from pathlib import Path

import pytest
from pydantic import ValidationError

from src.movie_predictor.api.config import (
    create_and_validate_config,
    fetch_config_from_yaml,
)

TEST_CONFIG_TEXT = """
package_name: movie_predictor
cleaned_data: cleaned_data.csv
drop_features: ['name', 'year', 'score']
target: score
test_size: 0.2
"""

def test_fetch_config_structure(tmpdir):  # tmpdir is pytest built-in fixture
    # Given
    # We make use of the pytest built-in tmpdir fixture
    configs_dir = Path(tmpdir)
    sample_config = configs_dir / "sample_config.yml"
    sample_config.write_text(TEST_CONFIG_TEXT)
    parsed_config = fetch_config_from_yaml(cfg_path=sample_config)

    # When
    config = create_and_validate_config(parsed_config=parsed_config)

    # Then
    assert config.model_config
    assert config.app_config


def test_config_validation_raises_error_for_invalid_config(tmpdir):
    # Given
    # We make use of the pytest built-in tmpdir fixture
    configs_dir = Path(tmpdir)
    sample_config = configs_dir / "sample_config.yml"

    # invalid config attempts to set a prohibited loss
    # function which we validate against an allowed set of
    # loss function parameters.
    sample_config.write_text(INVALID_TEST_CONFIG_TEXT)
    parsed_config = fetch_config_from_yaml(cfg_path=sample_config)

    # When
    with pytest.raises(ValidationError) as e_info:
        create_and_validate_config(parsed_config=parsed_config)

    # Then
    assert "not in the allowed set" in str(e_info.value)


# removed test_missing_config_field_raises_validation_error(tmpdir) with "field required" and "pipeline_name" 
