import sys
from pathlib import Path
from ml.regime import RegimeClassifier

def main():
    print("Verifying RegimeClassifier models...")
    rc = RegimeClassifier()
    if not rc._is_trained:
        print("ERROR: Model is not trained or could not be loaded.")
        sys.exit(1)
        
    print(f"Model loaded successfully. Found classes: {list(rc._dt.classes_)}")
    
    # Test a calm regime
    label_calm = rc.classify(volatility=0.0005, spread=0.0, trend_strength=0.999, volume=100.0)
    print(f"Test case 1 (calm) output: {label_calm}")
    
    # Test a volatile regime
    label_vol = rc.classify(volatility=0.002, spread=0.0, trend_strength=1.0, volume=1000.0)
    print(f"Test case 2 (volatile) output: {label_vol}")

    # Check if the outputs are in the valid set
    valid_regimes = set(rc._dt.classes_)
    for label in [label_calm, label_vol]:
        if label not in valid_regimes:
            print(f"WARNING: Output {label} is not one of the expected regimes {valid_regimes}")

    print("Verification complete.")

if __name__ == "__main__":
    main()
