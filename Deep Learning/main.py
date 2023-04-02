
import pathlib
from data_processing import matlab_data_extraction

def main():
    current_path = str(pathlib.Path(__file__).parent)
    signal_name = f'{current_path}/data/signal_data.mat'
    constellation_data, y_original, y_with_noise, channel_matrix  \
        = matlab_data_extraction(signal_name)

if __name__ == "__main__":
    
    main()