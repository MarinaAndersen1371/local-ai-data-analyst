from local_ai_data_analyst.analysis.analysis_service import analyze_dataset
from local_ai_data_analyst.analysis.schemas import DatasetAnalysis


def test_analyze_dataset_returns_validated_model(sample_dataframe):
    analysis = analyze_dataset(sample_dataframe)

    assert isinstance(analysis, DatasetAnalysis)


def test_analyze_dataset_calculates_total_revenue(sample_dataframe):
    analysis = analyze_dataset(sample_dataframe)

    assert analysis.sales.total_revenue == 550.0


def test_analyze_dataset_identifies_best_selling_product(sample_dataframe):
    analysis = analyze_dataset(sample_dataframe)

    assert analysis.sales.best_selling_product.product_name == "Laptop"
    assert analysis.sales.best_selling_product.units_sold == 5


def test_analyze_dataset_calculates_average_revenue_per_unit(sample_dataframe):
    analysis = analyze_dataset(sample_dataframe)

    averages = analysis.revenue_breakdown.average_revenue_per_unit

    assert averages["Laptop"] == 100.0
    assert averages["Notebook"] == 10.0


def test_analyze_dataset_identifies_monthly_revenue_extremes(sample_dataframe):
    analysis = analyze_dataset(sample_dataframe)

    assert analysis.monthly_revenue_extremes.highest.month == "2025-02"
    assert analysis.monthly_revenue_extremes.highest.revenue == 300.0

    assert analysis.monthly_revenue_extremes.lowest.month == "2025-01"
    assert analysis.monthly_revenue_extremes.lowest.revenue == 250.0
