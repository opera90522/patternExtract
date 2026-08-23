import csv

from patgen.io_csv import detect_text_column, read_texts


def write_csv(path, rows, fieldnames, encoding="utf-8"):
    with open(path, "w", newline="", encoding=encoding) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_detects_named_text_column():
    header = ["id", "body", "sender"]
    rows = [["1", "Your OTP is 1234", "BANK"]]
    assert detect_text_column(header, rows) == "body"


def test_falls_back_to_longest_column():
    header = ["id", "payload"]
    rows = [["1", "a fairly long sms message about a purchase"]]
    assert detect_text_column(header, rows) == "payload"


def test_reads_multiple_files_and_encodings(tmp_path):
    write_csv(tmp_path / "one.csv", [{"text": "hello"}], ["text"])
    write_csv(
        tmp_path / "two.csv", [{"text": "رصيدك"}], ["text"], encoding="cp1256"
    )
    texts = list(read_texts([str(tmp_path / "*.csv")]))
    assert sorted(texts) == sorted(["hello", "رصيدك"])


def test_limit_stops_early(tmp_path):
    write_csv(
        tmp_path / "c.csv",
        [{"text": f"msg {i}"} for i in range(10)],
        ["text"],
    )
    assert len(list(read_texts([tmp_path / "c.csv"], limit=3))) == 3
