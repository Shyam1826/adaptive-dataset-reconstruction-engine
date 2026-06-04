import pandas as pd
import argparse
import sys
from src.engine import AdaptiveReconstructionEngine

def main():
    parser = argparse.ArgumentParser(description="Adaptive Dataset Reconstruction Engine")
    parser.add_argument("input", help="Path to messy dataset (CSV/Excel)")
    parser.add_argument("--output", "-o", help="Path to save cleaned CSV", default="cleaned_output.csv")
    
    args = parser.parse_args()
    
    print(f"Loading {args.input}...")
    try:
        if args.input.endswith('.csv') or args.input.endswith('.txt'):

            df = pd.read_csv(
                args.input,
                sep=r'[\t,;|]',
                engine="python",
                header=None
            )

        else:
            df = pd.read_excel(args.input, header=None)

    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)
        
    print(f"Loaded raw dataset with shape {df.shape}")
    
    engine = AdaptiveReconstructionEngine(df)
    print("Running Adaptive Pipeline...")
    result = engine.run()
    
    if result is None:
        print("Failed to reconstruct the dataset. No valid strategy could be applied.")
        sys.exit(1)
        
    print("\n" + "="*50)
    print("RECONSTRUCTION COMPLETE")
    print("="*50)
    print(f"Best Strategy Selected : {result.strategy_name}")
    print(f"Confidence Score       : {result.confidence_score:.2f} / 100")
    print(f"Validation Score       : {result.validation_score:.2f} / 100")
    print("\nDataset Profile & Validation Info:")
    rep = result.validation_report
    print(f" - Rows                  : {rep.get('row_count')}")
    print(f" - Columns               : {rep.get('col_count')}")
    print(f" - Null Ratio            : {rep.get('null_ratio', 0):.2%}")
    print(f" - Numeric Columns Found : {rep.get('numeric_col_count')}")
    pipe = rep.get('pipeline_confidence', {})
    if pipe:
        print(f" - Pipeline Confidence     : {pipe.get('global_confidence', 0):.2%}")
        layers = pipe.get('layers', {})
        if layers:
            print("   Layer scores:", ", ".join(f"{k}={v:.2f}" for k, v in layers.items()))
    print("\nPreview of Reconstructed Data:")
    print("-" * 50)
    print(result.dataframe.head(10))
    print("-" * 50)
    
    engine.export(result, args.output)

if __name__ == "__main__":
    main()
