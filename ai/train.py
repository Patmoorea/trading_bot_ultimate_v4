from pandas import DataFrame, Series, read_csv, to_numeric
import argparse
import logging
from pathlib import Path
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def main(data_dir, epochs, batch_size, file_name='btc_usdc_1h_clean.csv'):
    try:
        # Chemin complet du fichier
        file_path = Path(data_dir) / file_name
        logger.info(f"Chargement des données depuis {file_path}")
        data = read_csv(file_path)
        # [Votre logique d'entraînement ici...]
        logger.info(f"Données chargées avec succès. Shape: {data.shape}")
    except Exception as e:
        logger.error(f"Erreur: {str(e)}")
        raise
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/historical')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--file', default='btc_usdt_1h_clean.csv',
                        help="Nom du fichier de données")
    args = parser.parse_args()
    main(args.data_dir, args.epochs, args.batch_size, args.file)
