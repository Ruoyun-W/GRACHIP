import argparse

from tools.data_generation import *

# from tools.DNA_encoder import *
from tools.fetch_genomic_data import *
from tools.hidden_state_from_DNABERT import *
from tools.mics import *
from tools.plot import *
from tools.seq_to_kmer import *
from model.grachip import *

def main():
    parser = argparse.ArgumentParser(description="Argument parser for grachip")

    # Adding named arguments
    parser.add_argument('--genomic_data_paths', default="./data/bwlist.txt", help="Paths to genomic files")
    parser.add_argument('--output_path', default="./output", help="Output directory")
    parser.add_argument('--genome_path', default="./data/hg38.fa", help="Reference genome .fa path")
    parser.add_argument('--model', default="./data/model.pt", help="Model path")
    parser.add_argument('--encoder', default="./data/encoder.pt", help="Encoder path")
    parser.add_argument('--input_edge', default="./data/input.cool", help="Input edge path")
    parser.add_argument('--start', required=True, type=int, help="Start site")
    parser.add_argument('--chrom', required=True, help="Chromosome")
    parser.add_argument('--DNABERT', default="./data/6-new-12w-0", help="DNABERT path")
    parser.add_argument('--iterations', default="10", type=int, help="Number of iterations for prediction")

    arg = parser.parse_args()

    sample_number_bins = 200

    bin_size = 10000
    print("generating genomic regions...")
    df_regions = create_region_dataframe(arg.chrom, arg.start, sample_number_bins)
    print("Extracting genomic signals...")
    genomic_signal_df = fetch_genomic_data(arg.genomic_data_paths,df_regions)
    print("Converting sequences to 6-mers...")
    kmer_DNA_path = regions2kmers(df_regions,arg.genome_path,6,arg.output_path)
    print("Getting DNA representations...")
    absolute_path = os.path.abspath(kmer_DNA_path)
    DNA_matrix = get_DNA_hidden_state(arg.DNABERT,absolute_path)
    print("Assembling dataset...")
    data = create_one_data(genomic_signal_df,arg.input_edge,DNA_matrix,df_regions)
    print("Start Predicting...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = os.path.join(arg.output_path,f"{arg.chrom}_{arg.start}")

    predictions = []

    for i in range(arg.iterations):
        print(f"Predicting... (Iteration {i+1}/{arg.iterations})")
        pred = predict(arg.model, data, device, save_path=None)
        predictions.append(pred)

    avg_pred = np.mean(predictions, axis=0)
    np.save(save_path + ".npy", avg_pred)
    print(f"Saved averaged predictions to {save_path}.npy")

    text = f'{df_regions["chrom"].iloc[0]} {df_regions["start"].iloc[0]/1000000}MB-{df_regions["end"].iloc[-1]/1000000}MB'
    print("Ploting...")

    plot(avg_pred,text=text,y="prediction",save_path=save_path+".png")
    print(f"Saved plot to {save_path}.png")

    return

if __name__ == '__main__':
    main()