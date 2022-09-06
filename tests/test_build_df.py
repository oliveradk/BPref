import os

import pytest
import build_df

class TestBuildDfUtils:

    def test_teacher_selection_filter(self):
        x = "fooo_bar_bazz_sample1_teacher_selectionuniform_state_maskN"
        assert build_df.teacher_selection_filter(x) == "uniform"
        
        x = "fooo_bar_bazz_sample1_teacher_selectionmax_beta_state_maskN"
        assert build_df.teacher_selection_filter(x) == "max_beta"
    
    def test_query_sample_filter(self):
        x = "fooo_bar_bazz_sample1_teacher_selectionuniform_state_maskN"
        assert build_df.query_sample_filter(x) == "1"

        x = "fooo_bar_bazz_sampleNone_teacher_selectionuniform_state_maskN"
        assert build_df.query_sample_filter(x) == "None"
