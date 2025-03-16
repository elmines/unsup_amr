import os
import torch
from .cli import CustomCLI
from. predict_mods import PredictMod
from .data_mods import EuroParlDataModule, AMRDataModule
from .constants import LIGHTNING_LOGS_DIR, CHECKPOINT_DIR

class PredictCLI(CustomCLI):

    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser=parser)
        parser.add_argument('--output_path', type=str, required=True, help="Path to save predictions.")

    
    def after_instantiate_classes(self):
        args = self.parser.parse_args()
        print("Parsed Arguments:", args)  # Debugging step
        self.output_path = args.output_path
        print(f"Predictions will be saved to: {self.output_path}")


def get_best_checkpoint_filepath():
    lightning_logs_path = os.path.join(os.path.dirname(__file__), os.pardir, LIGHTNING_LOGS_DIR)
    latest_version_dirs = sorted(os.listdir(lightning_logs_path), reverse=True)
    for version_dir in latest_version_dirs:
        for _, dir_names, _ in os.walk(os.path.join(lightning_logs_path, version_dir)):
            if CHECKPOINT_DIR in dir_names:
                checkpoint_dir_path = os.path.join(lightning_logs_path, version_dir, CHECKPOINT_DIR)
                if os.listdir(checkpoint_dir_path):
                    return os.path.join(checkpoint_dir_path, sorted(os.listdir(checkpoint_dir_path), reverse=True)[0])


if __name__ == "__main__":
    cli = PredictCLI(
        model_class = PredictMod,
        datamodule_class = AMRDataModule,
        trainer_defaults = {
            "max_epochs": 1,   #setting it 1 for now
            },
        run = False,   
        )

    best_checkpoint = torch.load(get_best_checkpoint_filepath())
    if best_checkpoint:
        print(f'weights before checkpoint load, {cli.model.embeddings.weight}')
        matching_state_dict = {k: v for k, v in best_checkpoint['state_dict'].items() if k in cli.model.state_dict()}
        cli.model.load_state_dict(matching_state_dict)
        print(f'weights after checkpoint load, {cli.model.embeddings.weight}')
        print(f't2a lm_head weights, {cli.model.t2a.lm_head.weight}')

    predictions = cli.trainer.predict(model=cli.model, datamodule=cli.datamodule)

    with open(cli.output_path, "w") as f:
        for prediction in predictions:
            for pred in prediction:
                f.write(str(pred) + "\n")
    
    print(f"Predictions saved to {cli.output_path}")
