from .cli import CustomCLI
from. predict_mods import PredictMod
from .data_mods import EuroParlDataModule


if __name__ == "__main__":

    class PredictCLI(CustomCLI):

        def add_arguments_to_parser(self, parser):
            parser.add_argument('--output_path', type=str, required=True, help="Path to save  predictions.")

        
        def after_instantiate_classes(self):
            self.output_path = self.parser.parse_args().output_path
            print(f"Predictions will be saved to: {self.output_path}")
        

    
    cli = CustomCLI(
        model_class = PredictMod,
        datamodule_class = EuroParlDataModule,
        trainer_defaults = {
            "max_epochs": 1,   #setting it 1 for now
            },
        run = False,   
        )
    
    predictions = cli.trainer.predict(model=cli.model, datamodule=cli.datamodule)

    with open(cli.output_path, "w") as f:
        for prediction in predictions:
            for pred in prediction:
                f.write("".join(pred) + "\n")
    
    print(f"Predictions saved to {cli.output_path}")
