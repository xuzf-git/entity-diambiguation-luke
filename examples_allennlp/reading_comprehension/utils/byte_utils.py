from typing import Tuple, List


def byte_str(text: str) -> bytes:
    return text.encode("utf-8")


def byte_len(text: str) -> int:
    return len(text.encode("utf-8"))


def byte_slice(text: str, start: int, end: int, errors="replace") -> str:
    # Python 3 encodes text as character sequences, not byte sequences
    # (like Python 2).
    return byte_str(text)[start:end].decode("utf-8", errors=errors)


def byte_to_char_offset(text: str, byte_offset: int) -> int:
    return len(byte_str(text)[:byte_offset].decode("utf-8", errors="replace"))


def char_to_byte_offset(text: str, char_offset: int) -> int:
    return len(byte_str(text[:char_offset]))


def get_token_to_plaintext_byte_mapping(
    tokens: List[str], token_to_contexts_byte_mapping: List[Tuple[int, int]], context_to_plaintext_offset: List[int],
) -> List[Tuple[int, int]]:

    # sometimes token_to_contexts_byte_mapping contains out of range index probably for some special characters
    # append the dummy mapping -1 here
    context_to_plaintext_offset.append(-1)

    token_to_plaintext_byte_mapping = [
        (context_to_plaintext_offset[s], context_to_plaintext_offset[e]) if (s, e) != (-1, -1) else (-1, -1)
        for (s, e) in token_to_contexts_byte_mapping
    ]

    # sometimes the mapping fails to track the index,
    # for example, when a entire paragraph is deleted from the plaintext
    # So fix that here.
    for i, (t, (s, e)) in enumerate(zip(tokens, token_to_plaintext_byte_mapping)):
        if s != -1 and e == -1:
            token_to_plaintext_byte_mapping[i] = (s, s + byte_len(t))

    return token_to_plaintext_byte_mapping
