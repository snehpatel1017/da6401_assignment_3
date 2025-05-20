import wandb
from .prediction_vanilla import log_test_predictions
from .Basic_Model import train_model
# best model config
class Config:
    def __init__(self,
                 embed_dim=64,
                 hidden_dim=256,
                 num_layers=3,
                 cell_type="LSTM",
                 dropout=0.2,
                 lr=0.001,
                 batch_size=32,
                 lang_code="hi",
                 epochs=10):          # note 'epochs' not 'epoch'
        self.embed_dim   = embed_dim
        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers
        self.cell_type   = cell_type
        self.dropout     = dropout
        self.lr          = lr
        self.batch_size  = batch_size
        self.lang_code   = lang_code
        self.epochs      = epochs

config = Config()

# wandb.login(key="2b8654ea1d7143307fd59d1ea1bda5bc9f6fef77")
# wandb.init(project="da6401_assignment_3", entity="cs24m048-iit-madras" , name="test_prediction_grid_without_attention")
model , dataset , test_loader , device = train_model(config)
# best_table = log_test_predictions(model, dataset, test_loader, device, n_samples=len(test_loader))