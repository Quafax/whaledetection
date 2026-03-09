from huggingface_hub import login
login()
from whaledetection.config.config_loader import load_config
from whaledetection.datasets.wmms_database import export_wmms_database
from whaledetection.datasets.whaleFM_database import export_whalefm_database
#Could do a registry for all the databases but for just the few i like it simpler. 

cfg = load_config("configs/config.yaml")
database_base_dir_out = cfg.loadDatabase.database_base_dir_out

#database load functions
def export_watkins():
    export_wmms_database(database_base_dir_out)

def export_whaleFM():
    export_whalefm_database(database_base_dir_out)


if __name__ == "__main__":

    #load the databases here. The loading is not fully automated on purpose so that modifications can be made
    #export_whaleFM()
    #export_watkins()
    pass