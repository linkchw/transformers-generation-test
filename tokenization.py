import regex as re
import os , json
import tiktoken 


text = "گُنْجـِشْکْ به پرندگان خانوادهٔ گنجشکان گفته می‌شود، که خود در راستهٔ گنجشک‌سانان می‌گنجند. گنجشک خانگی معروف‌ترین گونهٔ گنجشک است که در تعداد بسیار زیاد در همۀ قاره‌های جهان و بیشتر در مناطق شهری زندگی می‌کند. گنجشک پرنده‌ای است کوچک با نوک کلفت و مخروطی و پاهای نسبتاً کوتاه که اغلب پر و بال رنگارنگ ندارد. گنجشک‌ها بیشتر به‌صورت اجتماعی به سر می‌برند و به‌طور گروهی زادوولد می‌کنند. در بیشتر گونه‌ها دودیسی جنسی (تفاوت نر و ماده) دیده می‌شود. در سوراخ‌های درختان، شکاف دیوار و صخره‌ها لانه می‌سازند. بیشتر از دانه‌ها و مواد گیاهی و گاه جانوری تغذیه می‌کنند. معمولاً گنجشک‌هایی که در شهرها زندگی می‌کنند از نوع گنجشک خانگی و گنجشک درختی هستند."
tokens = text.encode("utf-8") # raw bytes
tokens = list(map(int, tokens)) # convert to a list of integers in range 0..255 for convenience
print('---')
print(text)
print("length:", len(text))
print('---')
print(tokens)
print("length:", len(tokens))


# Get stats
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts



# Merge
def merge(ids, pair, idx):
  newids = []
  i = 0
  while i < len(ids):
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids


# stats = get_stats(tokens)
# top_pair = max(stats, key=stats.get)

# print(merge([5, 6, 6, 7, 9, 1], (6, 7), 99))
# tokens2 = merge(tokens, top_pair, 256)
# print(tokens2)
# print("lenght: ", len(tokens2))

# ---
vocab_size = 276 # the desired final vocabulary size
num_merges = vocab_size - 256
ids = list(tokens) # copy so we don't destroy the original list

merges = {} # (int, int) -> int
for i in range(num_merges):
    try:
        stats = get_stats(ids)
        pair = max(stats, key=stats.get) 
        idx = 256 + i
        print(f"merging {pair} into a new token {idx}")
        ids = merge(ids, pair, idx)
        merges[pair] = idx
    except ValueError as e :
        print(e)

print("token lenght: ", len(tokens))
print("ids lenght: ", len(ids))
print(f"comprasion ratio: {len(tokens) / len(ids):.2f}X")


vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1),  idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

def decode(ids):
    # given ids (list of integers), return Python string 
    tokens  = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text

print(decode([128]))



def encode(text):
    # given a string, return list of integers (the tokens)
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break

        idx = merges[pair] 
        tokens = merge(tokens, pair, idx)
    return tokens

print(encode("علیرضا"))
print(decode(encode("علیرضا")))

print(encode("Hello, world!"))
print(decode(encode("Hello, world!")))

gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
print(re.findall(gpt2pat, "Hello, world!"))


# enc = tiktoken.get_encoding("gpt2")
# print(enc.encode("علیرضا"))
# print(enc.decode(enc.encode("علیرضا")))

