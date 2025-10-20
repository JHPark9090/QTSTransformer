from braindecode.models import EEGNeX
import torch

from pathlib import Path

def resolve_path(name="python_packages"):
    if Path(f"/app/input/res/{name}").exists():
        return f"/app/input/res/{name}"
    elif Path(f"/app/input/{name}").exists():
        return f"/app/input/{name}"
    elif Path(f"{name}").exists():
        return f"{name}"
    elif Path(__file__).parent.joinpath(f"{name}").exists():
        return str(Path(__file__).parent.joinpath(f"{name}"))
    else:
        raise FileNotFoundError(
            f"Could not find {name} folder in /app/input/res/ or /app/input/ or current directory"
        )

class ModelWithExtraDeps(torch.nn.Module):
    def __init__(self, challenge):
        super().__init__()
        import sys
        sys.path.append(resolve_path())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from QuixerTSModel_Pennylane2 import QuixerTimeSeriesPennyLane
        
        if challenge==1:
            self.real_model = QuixerTimeSeriesPennyLane(
                sim14_circuit_config=['RX', 'CRY', 'RX', 'CRY_counter'],
                feature_projection_layer='Conv2d_GLU',
                output_ff_layer='GLU',
                n_qubits=6,
                n_timesteps=200,
                degree=2,
                n_ansatz_layers=2,
                feature_dim=129,
                output_dim=1,
                dropout=0.1,
                device=device
            )
        elif challenge==2:
            self.real_model = QuixerTimeSeriesPennyLane(
                sim14_circuit_config=['RY', 'IsingXX', 'RY', 'IsingXX_counter'],
                feature_projection_layer='Conv2d_GLU',
                output_ff_layer='GLU',
                n_qubits=8,
                n_timesteps=400,
                degree=3,
                n_ansatz_layers=2,
                feature_dim=129,
                output_dim=1,
                dropout=0.1,
                device=device
            )            
        self.real_model.to(device).float()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.real_model(x)

class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE

    def get_model_challenge_1(self):
        model_challenge1 = ModelWithExtraDeps(challenge=1).to(self.device)
        # Load trained weights for Challenge 1
        model_path = Path(__file__).parent / "weights_challenge_1_externalizing_B1.pt"
        if model_path.exists():
            print("Loading weights for Challenge 1 model.")
            model_challenge1.real_model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        else:
            print("Warning: No 'weights_challenge_1_externalizing_B1.pt' found. Using an untrained model for Challenge 1.")         
        return model_challenge1

    def get_model_challenge_2(self):
        model_challenge2 = ModelWithExtraDeps(challenge=2).to(self.device)
        # Load trained weights for Challenge 2
        model_path = Path(__file__).parent / "weights_challenge_2_C2_mini3.pt"
        if model_path.exists():
            print("Loading weights for Challenge 2 model.")
            model_challenge2.real_model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        else:
            print("Warning: No 'weights_challenge_2_C2_mini3.pt' found. Using an untrained model for Challenge 2.")        
        return model_challenge2
