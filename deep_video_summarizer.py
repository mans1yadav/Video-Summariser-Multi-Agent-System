import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional
import cv2

class VideoSummarizationModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        # LSTM for temporal feature extraction
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Transformer encoder for attention-based frame selection
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim * 2,  # * 2 for bidirectional LSTM
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Final layers for importance score prediction
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Transformer processing with optional masking
        transformer_out = self.transformer(lstm_out.transpose(0, 1), src_key_padding_mask=mask)
        transformer_out = transformer_out.transpose(0, 1)
        
        # Final prediction layers
        out = self.fc1(transformer_out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        scores = self.sigmoid(out)
        
        return scores.squeeze(-1)

class VideoDataset(Dataset):
    def __init__(self, video_path: str, feature_extractor, segment_length: int = 16):
        self.video_path = video_path
        self.feature_extractor = feature_extractor
        self.segment_length = segment_length
        self.frames, self.frame_count = self._load_video()
        
    def _load_video(self) -> Tuple[List[np.ndarray], int]:
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames, len(frames)
    
    def __len__(self) -> int:
        return (self.frame_count - 1) // self.segment_length + 1
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        start_idx = idx * self.segment_length
        end_idx = min(start_idx + self.segment_length, self.frame_count)
        
        segment_frames = self.frames[start_idx:end_idx]
        if len(segment_frames) < self.segment_length:
            # Pad with zeros if needed
            padding = [np.zeros_like(segment_frames[0]) for _ in range(self.segment_length - len(segment_frames))]
            segment_frames.extend(padding)
        
        # Extract features using the provided feature extractor
        features = []
        for frame in segment_frames:
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frame_tensor = frame_tensor.unsqueeze(0)
            with torch.no_grad():
                feature = self.feature_extractor(frame_tensor)
            features.append(feature.squeeze(0))
        
        return torch.stack(features)

class DeepVideoSummarizer:
    def __init__(self, model_config: dict = None):
        if model_config is None:
            model_config = {
                'input_dim': 2048,  # ResNet feature dimension
                'hidden_dim': 512,
                'num_layers': 2,
                'num_heads': 8,
                'dropout': 0.1
            }
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = VideoSummarizationModel(**model_config).to(self.device)
        self.feature_extractor = self._init_feature_extractor()
        
    def _init_feature_extractor(self) -> nn.Module:
        # Use ResNet-50 as feature extractor
        resnet = torchvision.models.resnet50(pretrained=True)
        feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        feature_extractor.eval()
        return feature_extractor.to(self.device)
    
    async def summarize_video(self, video_path: str, summary_ratio: float = 0.3) -> List[int]:
        """
        Generate video summary by selecting key frames
        
        Args:
            video_path: Path to input video file
            summary_ratio: Desired ratio of frames to keep in summary
            
        Returns:
            List of selected frame indices for the summary
        """
        try:
            # Prepare dataset and dataloader
            dataset = VideoDataset(video_path, self.feature_extractor)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            
            # Get importance scores for all segments
            all_scores = []
            with torch.no_grad():
                for batch in dataloader:
                    batch = batch.to(self.device)
                    scores = self.model(batch)
                    all_scores.extend(scores.cpu().numpy())
            
            # Select frames based on importance scores
            all_scores = np.array(all_scores)
            num_frames_to_keep = int(len(all_scores) * summary_ratio)
            selected_indices = np.argsort(all_scores)[-num_frames_to_keep:]
            selected_indices.sort()
            
            return selected_indices.tolist()
            
        except Exception as e:
            logger.error(f"Video summarization failed: {str(e)}")
            raise
            
    def save_summary(self, video_path: str, output_path: str, selected_frames: List[int]):
        """Save selected frames as a summary video"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_idx in selected_frames:
                    out.write(frame)
                frame_idx += 1
            
            cap.release()
            out.release()
            
        except Exception as e:
            logger.error(f"Failed to save summary video: {str(e)}")
            raise