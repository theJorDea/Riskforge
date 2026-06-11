"""Optional ML extensions (require: pip install -e ".[ml]")."""

from riskforge.ml.lstm_vol import lstm_vol_fit, lstm_vol_predict

__all__ = ["lstm_vol_fit", "lstm_vol_predict"]
