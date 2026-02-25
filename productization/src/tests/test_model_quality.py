from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy import stats as statspy

ALPHA = 0.05
BENCHMARK_PATH = Path(__file__).parent / "data" / "benchmark_predictions.csv"
REQUIRED_COLUMNS = {"y_true", "y_pred_old", "y_pred_new"}


@pytest.fixture(scope="module")
def benchmark_df() -> pd.DataFrame:
    assert BENCHMARK_PATH.exists(), (
        f"Arquivo de benchmark não encontrado em: {BENCHMARK_PATH}. "
        "Crie o arquivo com colunas: y_true, y_pred_old, y_pred_new."
    )

    df = pd.read_csv(BENCHMARK_PATH)
    missing = REQUIRED_COLUMNS - set(df.columns)
    assert not missing, f"Colunas ausentes no benchmark: {sorted(missing)}"
    assert len(df) > 30, "Benchmark muito pequeno para teste estatístico robusto."

    return df


def _absolute_error(df: pd.DataFrame, pred_col: str) -> np.ndarray:
    return (df["y_true"] - df[pred_col]).abs().to_numpy()


def test_novo_modelo_melhor_que_antigo_com_kolmogorov_smirnov(benchmark_df: pd.DataFrame) -> None:
    """
    H1: erro_novo é estocasticamente menor que erro_antigo.
    Em ks_2samp, isso equivale a CDF(erro_novo) > CDF(erro_antigo),
    portanto alternative='greater' com data1=erro_novo.
    """
    err_new = _absolute_error(benchmark_df, "y_pred_new")
    err_old = _absolute_error(benchmark_df, "y_pred_old")

    ks_result = statspy.ks_2samp(err_new, err_old, alternative="greater", method="auto")

    assert ks_result.pvalue < ALPHA, (
        f"Sem evidência estatística de melhora do novo modelo (p={ks_result.pvalue:.6f}, alpha={ALPHA})."
    )
    assert ks_result.statistic > 0, (
        "Estatística KS não indica dominância do novo modelo."
    )


def test_novo_modelo_tem_menor_erro_medio_no_benchmark(benchmark_df: pd.DataFrame) -> None:
    err_new = _absolute_error(benchmark_df, "y_pred_new")
    err_old = _absolute_error(benchmark_df, "y_pred_old")

    assert err_new.mean() < err_old.mean(), (
        f"Erro médio do novo modelo ({err_new.mean():.6f}) "
        f"não foi menor que o do antigo ({err_old.mean():.6f})."
    )