import pandas as pd
import pytest
from src.services.data_loader import load_faq_data

def test_load_faq_data_success(mock_load_faq_data_success):
    mock_file, mock_exists = mock_load_faq_data_success
    load_faq_data.cache_clear()
    df = load_faq_data("dummy_path.json")
    assert not df.empty
    assert len(df) == 3
    assert df.loc[0, "question"] == "Q1"
    assert "id" in df.columns
    mock_file.assert_called_once_with("dummy_path.json", "r", encoding="utf-8")

def test_load_faq_data_file_not_found(): 
    load_faq_data.cache_clear()
    df = load_faq_data("non_existent_file.json")
    assert df.empty

def test_load_faq_data_invalid_json(mock_load_faq_data_invalid_json):
    mock_file = mock_load_faq_data_invalid_json
    load_faq_data.cache_clear()
    df = load_faq_data("invalid.json")
    assert df.empty
    mock_file.assert_called_once_with("invalid.json", "r", encoding="utf-8")

def test_load_faq_data_missing_id_column(mock_load_faq_data_no_id):
    mock_file = mock_load_faq_data_no_id
    load_faq_data.cache_clear()
    df = load_faq_data("no_id.json")
    assert df.empty
    mock_file.assert_called_once_with("no_id.json", "r", encoding="utf-8")