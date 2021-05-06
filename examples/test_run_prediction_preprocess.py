from run_prediction_preprocess import update_variant

seq = "DRPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPR"
wt_seq = "DVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPR"
variant = "V171R"

deleted_cols = [0, 1]
expected = None
actual = update_variant(seq, wt_seq, variant, deleted_cols)
assert expected == actual

deleted_cols = [0, 2]
expected = "V1R"
actual = update_variant(seq, wt_seq, variant, deleted_cols)
assert expected == actual, "Expected " + expected + ", but given " + actual

print("OK.")