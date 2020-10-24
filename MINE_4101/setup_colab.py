import setup_colab_general as setup_general

def setup_lab5():
    setup_general.setup_general()
    from utils import general as gen
    id_insurance = "11jaAMuHLypta8BXyUfPPOnql-PXzLQGD"
    gen.download_file_from_google_drive(id_insurance, "insurance.csv")
    id_wine = "1Je03icLBNGad8q58QnJ-eQKex82t3exP"
    gen.download_file_from_google_drive(id_wine, "winequality.csv")