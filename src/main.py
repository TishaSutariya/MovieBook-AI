from preprocess import run_preprocessing
from eda import run_eda


def main():

    print("\n")
    print("=" * 70)
    print("MOVIEBOOK AI")
    print("Movie + Book Recommendation System")
    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1: Preprocessing
    # --------------------------------------------------------

    print("\nSTEP 1: PREPROCESSING")

    run_preprocessing()

    # --------------------------------------------------------
    # STEP 2: Exploratory Data Analysis
    # --------------------------------------------------------

    print("\nSTEP 2: EXPLORATORY DATA ANALYSIS")

    run_eda()

    # --------------------------------------------------------
    # Completed
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()