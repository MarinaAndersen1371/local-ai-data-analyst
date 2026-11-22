from unittest.mock import patch

from local_ai_data_analyst.logging_config import setup_logging


def test_setup_logging_configures_info_level():
    with patch(
        "local_ai_data_analyst.logging_config.logging.basicConfig"
    ) as mock_config:
        setup_logging()

    mock_config.assert_called_once()

    config = mock_config.call_args.kwargs

    assert config["level"] == "INFO" or config["level"] == 20
    assert "%(asctime)s" in config["format"]
