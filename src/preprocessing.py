import re

VIETNAMESE_PATTERN = {
    "[àáảãạăắằẵặẳâầấậẫẩ]": "a",
    "[ÀÁẢÃẠĂẮẰẴẶẲÂẦẤẬẪẨ]": "A",
    "[đ]": "d",
    "[Đ]": "D",
    "[èéẻẽẹêềếểễệ]": "e",
    "[ÈÉẺẼẸÊỀẾỂỄỆ]": "E",
    "[ìíỉĩị]": "i",
    "[ÌÍỈĨỊ]": "I",
    "[òóỏõọôồốổỗộơờớởỡợ]": "o",
    "[ÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ]": "O",
    "[ùúủũụưừứửữự]": "u",
    "[ÙÚỦŨỤƯỪỨỬỮỰ]": "U",
    "[ỳýỷỹỵ]": "y",
    "[ỲÝỶỸỴ]": "Y",
}

def convert_accented_vietnamese_text(data: str, lower=True) -> str:
    if not data:
        return ""
    if lower is True:
        data = data.lower()
    for regex, replace in VIETNAMESE_PATTERN.items():
        data = re.sub(regex, replace, data)
    data = data.replace("\n", " ")
    return data