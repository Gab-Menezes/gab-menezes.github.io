---
layout: post
title: "Using the most unhinged AVX512 instruction to make the fastest phrase search algo"
---

# Disclaimers before we start
* The contents of this blog post are inspired by the wonderful idea of [Doug Turbull](https://softwaredoug.com/) from the series of blog posts about [Roaringish](https://softwaredoug.com/blog/2024/01/21/search-array-phrase-algorithm). In here we will take this ideas to an extreme, from smart algos to raw performance optimization.
* I highly recommend reading the [Roaringish](https://softwaredoug.com/blog/2024/01/21/search-array-phrase-algorithm) blog post, but if you don't want there will be a recap on how it works.
* This project it's been almost 7 months in the making, thousands and thousands of line of code have been written an rewritten, so bare with me if I sound crazy. At the moment of writing there is almost 2.7k LOC, but I have commited around 17k LOC (let's take a few thousands because of `.lock` files) (probably at the time of publishing this number has increased), so the project has be rewritten almost 6 times.

![](/assets/2025-01-13-using-the-most-unhinged-avx512-instruction-to-make-the-fastest-phrase-search-algo/loc.png)

* At the beginning I wasn't planning on writing a blog post about it, but I invested so much time and the end result is so cool that I think it's worth a blog post.
* I started writing a first version but didn't like the direction it was going, so I decided to change it and here we are. So my plan is to show the current state of the project and try my best to remember and to explain why things are the way they are, a lot of benchmarking and fine tuning has been taken place so it's almost impossible for me to remember everything. Maybe in some cases I will go back in time to explain the reason certain optimization was chosen and others I might just explain how it works. This post will probably be long, so grab some water/tea/coffee.
* Again every piece of code that you will get to see has been rewritten a lot of times, I didn't magically arrive at the solution.
* There will be a lot of `unsafe` keywords, don't be afraid of it.
* **We won't talk about**:
  * Multi threading, since this workload is trivially parallelizable (by sharding) it's kinda a easy to do it, and it's scales pretty much linearly, so if you are curious on who this would perform on `N` threads, just get the numbers and devide by `N`.
  * Things outside my code to make it faster, like enabling Huge Pages.
  * The main focus will be in the search part, so there will be no focus in the indexing part, only when needed
* **Benchmarking methodology**:
  * We are optimizing for raw latency, the lower the better.
  * I have randomly select a bunch of queries (each with different bottleneck profile) and measured how each change impacted it's performance.
  * I ran each query 20 times to warmup the CPU, after that I ran the same query another 1000 times and collected the time taken as a whole and in each crucial step, with this we have time/iter.
  * For those who say that that's not a good benchmark and I should have used criterion with statistical analysis and blah blah blah... All I can say is both of my system (spoiler alert) are pretty reproducible (up to 5 us on the whole benchmark, not per iter, per benchmark run !!!) if I run the same benchmark twice. So this is good enough.
  * So after the CPU is warm the time in each iteration is pretty damn consistent, so that's why I consider this good enough.
  * The dataset used is the same used by Doug, [MS MARCO](https://microsoft.github.io/msmarco/), containing 3.2M documents, around 22GB of data. It consists of a link, question and a long answer, in this case we only index the answer (so 20/22GB of data) (in the original article only 1M documents were used, but in here we ingest all of it). 
  * Getting close to the end there will be some benchmarks and to finalize there will be a comparision against [Meilisearch](https://www.meilisearch.com/) (production ready Search Engine, who is known for it's good performance).
  * Spec of both of my systems where I ran all of the benchmarks:
    * Notebook (where most of developemnt took part): i5-1135G7 - 16GB
    * Desktop (final results on this system): 9700x - 64GB (Spoiler)
* There will be a lot of source code for those who are intrested, but the unecessary ones will be collapsed, if you are not that intrested you can just skip those.
* Also a huge thanks to all of the people who helped me through this, specially the super kind people on the [Gamozo's discord](https://discord.gg/gwfvzzQC), how through the last year had to see me go crazy about intersection algos.
* **Why am I doing this ?** Because I like it and wanted to nerd snipe some people.


# What are we doing ? And why we care ?
Do you know when you go to your favorite search engine and search for something using double quotes, like passage of a book/article or something very specific ? That's called phrase search (sometimes exact search), what we are telling the search engine is that we want this exact words in this exact order (this varies from search engine to search engine, but that's the main idea). In contrast when searching by keywords (not in double quotes) we don't care about the order or if it's the exact word, it may be a variation.

There is one big difference between this two, searching by keywords is relative cheap when compared to doing a phrase search. In both cases we are calculating the intersection between the reserve indexes (in the keyword search we may take the union, but for the sake of simplicity let's assume the intersection), but in the phrase search we need to keep track where each token appeared to guarantee that we are looking for tokens close to each other.

So this makes phrase search way more computationally expensive, the convetional algo found in books is very slow, let's take a look at this first and compare with Doug's brilliant idea.

## How the conventional algo works
![](/assets/2025-01-13-using-the-most-unhinged-avx512-instruction-to-make-the-fastest-phrase-search-algo/convetional-phrase-search-algo.png)

This algo (taken from Information Retrieval: Implementing and Evaluating Search Engines) analyzes one document at the time, so the `nextPhrase` function needs to called for each document in the intersection of document ids in the index.

Description (also taken from the book): Locates the first occurrence of a phrase after a given position. The function calls the `next(t, i)` method which returns next position `t` after the position i. Similar to `prev(t, i)`.

You don't need to understand this algo, just that is inefficient because of a lot of reasons:
  * Intersection is expensive
  * Analyzes one document and the time
  * Not cache friendly we are jumping around with the `next` and `prev` functions
  * Recursive

For a small collection of documents this works fine, but imagine for large collections 1M+ documents this will blow up quiclky (hundreds of milliseconds, maybe even seconds per query), which is unacceptable for a real time search engine, so not good at all.


## The genius idea
How did Doug fixed this ? With a lot of clever bit hacking. The following example is taken from the blog post and summarized by me, also fixed some small mistakes in the examples (again highly recommend reading the [original blog post](https://softwaredoug.com/blog/2024/01/21/search-array-phrase-algorithm)).

Let's say we want to index the following documents to be able to phrase search them in the future
```
doc 0: "mary had a little lamb the lamb ate mary"
doc 1: "uhoh little mary dont eat the lamb it will get revenge"
doc 2: "the cute little lamb ran past the little lazy sheep"
doc 3: "little mary ate mutton then ran to the barn yard"
```

The inverted index will look something like this:
```yaml
mary:
   docs:
     - 0:
         posns: [0, 8]
     - 1:
         posns: [2]
little:
   docs:
     - 0:
         posns: [3]
     - 1:
         posns: [1]
     - 2:
         posns: [2, 7]
     - 3:
         posns: [0]
lamb:
   docs:
     - 0:
         posns: [4, 6]
     - 1:
         posns: [6]
     - 2:
         posns: [3]
...
```

To do a phrase search we can connect two terms at the time, the left and right one. Searching by "mary had a little lamb", will result in searching by:

* `"mary"` and `"had"` = `"mary had"`
* `"mary had"` and `"a"` = `"mary had a"`
* ...

So we reuse the work done the previous step, just connect the right term with the previous. Imagine the scenario where `"mary had"` occurs in the following positions: `[1, 6]` and `"a"` appers in the position `[2]`, so `"mary had a"` occurs in the positions `[2]`, we keep doing this for the next token until we finish it.

The main idea to recreate and optimize this behaviour was taken from [`Roaring Bitmaps`](https://roaringbitmap.org/), that's why it's called `Roaringish`. We want to pack as much data as possible and avoid storing the positions for each document for each term separatly.

Assuming that the `pos <= 2^16 * 16 = 1048576` (i.e the maximum document length is 1048576 tokens which is very reasonable, it's way more than [Meilisearch for example](https://www.meilisearch.com/docs/learn/resources/known_limitations#maximum-number-of-words-per-attribute), so this should be fine) allows us to decompose this value into two 16 bits, one representing the group and other the value, where `pos = group * 16 + value`

```python
posns  =              [1, 5, 20, 21, 31, 100, 340]
groups = posns // 16 #[0, 0,  1,  1,  1,   6,  21]
values = posns %  16 #[1, 5,  4,  5, 15,   4,   4]
```

Since `value < 15`, we can pack even more data by bitwising each group value into single 16 bit number

```
Group 0             Group 1            ...   Group 21
0000000000100010    1000000000110000   ...   0000000000010000
(bits 1 and 5 set)  (bits 4, 5, 15 set)      
```

With this we can pack the group and the values in a single 32 bit number, by shifting the group and or it with the packed values i.e `group_value = (group << 16) | values`.

```
Group 0                              | Group 1                               ...    | Group 21
0000000000000000 0000000000100010    | 0000000000000001 1000000000110000     ...    | 0000000000010101 0000000000010000
(group 0)        (bits 1 and 5 set)  | (group 1)        (bits 4, 5, 15 set)  ...    | (group 21)       (bit 4 set)
```

Now to find if the left token is followed by the right token we can:
1. Intersect the MSB's (group part)
2. `Shift left` the LSB bits from the left token by 1 (values part)
3. `And` the LSB's
4. If there is at least one bit set in the LSB's after the `and` the left token is followed by the right token

For example, let's assume the term "little" has the following positions.
```
Group 0                              ...    | Group 21
0000000000000000 0000000000100010    ...    | 0001000000100101 0000000000010100
```

And "lamb".
```
Group 1                              ...    | Group 21
0000000000000000 0000000000100010    ...    | 0000000000010101 0000000001001000
```

The group 21 is in the intersection, so:

`(0000000000010100 << 1) & 0000000001001000 = 0000000000001000`

With this "little lamb" is found in group 21 value 3, i.e `21*16+3 = 339`.

And the magical part is: To avoid having to analyze one document at the time we can pack the document id (assuming 32 bit document id) into the 32 MSB of a 64 bit number, while the LSB are the group and value. `packed = (doc_id << 32) | group_value`

When calculating the intersection we take the document id and group (48 MSB's), with this we can search the whole index in a single shot. So in the end we have a single continuous array of data, with all document ids and positions that contain that token.

Pretty f* cool.

**Note:** For those who are paying attention you might have spotted a problem, we will talk about this later.

# How my version works ?
The idea behind my implementation is kinda similiar, but with a lot of extra steps, here we have a super brief overview on how it all works, later we will explore each step in depth.

![](/assets/2025-01-13-using-the-most-unhinged-avx512-instruction-to-make-the-fastest-phrase-search-algo/diagram0.png)

Step `1` and `2` are here to help us reduce as much as possible the time spent on step `3`, since it's the most expensive one. So both of them try to be smart and find the best pattern for us to tackle the same problem, by reducing the search space.

Most of our time will be spent on step `3`, but step `1` is pretty cool.

{% details **Code** for the main body of the search function **(you can ignore if you want)** %}
```rust
pub fn search<I: Intersect>(
    &self,
    q: &str,
    common_tokens: &HashSet<Box<str>>,
    mmap: &Mmap,
) -> Vec<u32> {
    let tokens = Tokens::new(q);
    let tokens = tokens.as_ref();

    if tokens.is_empty() {
        return Vec::new();
    }

    let rotxn = self.env.read_txn().unwrap();
    if tokens.len() == 1 {
        return self
            .get_roaringish_packed(&rotxn, tokens.first().unwrap(), mmap)
            .map(|p| p.get_doc_ids(stats))
            .unwrap_or_default();
    }

    // Step 1
    let bump = Bump::with_capacity(tokens.reserve_len() * 5);
    let (final_tokens, token_to_packed) =
        self.merge_and_minimize_tokens(&rotxn, tokens, common_tokens, mmap, &bump);

    let Some(final_tokens) = final_tokens else {
        return Vec::new();
    };

    if final_tokens.next.is_none() {
        return token_to_packed
            .get(&final_tokens.tokens)
            .unwrap()
            .get_doc_ids(stats);
    }

    let final_tokens: Vec<_> = final_tokens.iter().copied().collect();

    // Step 2
    let mut min = usize::MAX;
    let mut i = usize::MAX;
    for (j, ts) in final_tokens.windows(2).enumerate() {
        let l0 = token_to_packed.get(&ts[0]).unwrap().len();
        let l1 = token_to_packed.get(&ts[1]).unwrap().len();
        let l = l0 + l1;
        if l <= min {
            i = j;
            min = l;
        }
    }

    let lhs = &final_tokens[i];
    let mut lhs_len = lhs.len() as u32;
    let lhs = token_to_packed.get(lhs).unwrap();

    let rhs = &final_tokens[i + 1];
    let mut rhs_len = rhs.len() as u32;
    let rhs = token_to_packed.get(rhs).unwrap();

    // Step 3
    let mut result = lhs.intersect::<I>(*rhs, lhs_len);
    let mut result_borrow = BorrowRoaringishPacked::new(&result);

    let mut left_i = i.wrapping_sub(1);
    let mut right_i = i + 2;

    // Step 2 and 3
    loop {
        let lhs = final_tokens.get(left_i);
        let rhs = final_tokens.get(right_i);
        match (lhs, rhs) {
            (Some(t_lhs), Some(t_rhs)) => {
                let lhs = token_to_packed.get(t_lhs).unwrap();
                let rhs = token_to_packed.get(t_rhs).unwrap();
                if lhs.len() <= rhs.len() {
                    lhs_len += t_lhs.len() as u32;

                    result = lhs.intersect::<I>(result_borrow, lhs_len);
                    result_borrow = BorrowRoaringishPacked::new(&result);

                    left_i = left_i.wrapping_sub(1);
                } else {
                    result = result_borrow.intersect::<I>(*rhs, rhs_len);
                    result_borrow = BorrowRoaringishPacked::new(&result);

                    lhs_len += rhs_len;
                    rhs_len = t_rhs.len() as u32;

                    right_i += 1;
                }
            }
            (Some(t_lhs), None) => {
                let lhs = token_to_packed.get(t_lhs).unwrap();
                lhs_len += t_lhs.len() as u32;

                result = lhs.intersect::<I>(result_borrow, lhs_len);
                result_borrow = BorrowRoaringishPacked::new(&result);

                left_i = left_i.wrapping_sub(1);
            }
            (None, Some(t_rhs)) => {
                let rhs = token_to_packed.get(t_rhs).unwrap();

                result = result_borrow.intersect::<I>(*rhs, rhs_len);
                result_borrow = BorrowRoaringishPacked::new(&result);

                lhs_len += rhs_len;
                rhs_len = t_rhs.len() as u32;

                right_i += 1;
            }
            (None, None) => break,
        }
    }

    // Step 4
    result_borrow.get_doc_ids()
}
```
{% enddetails %}
<br/> 
So you might have noticed that the intersection is composed of two phases, why is that ? There is an annoying issue with Roaringish is due to the edge case where value bits are in the boundry of the group and calculating the intersection would lead to an incorrect result (that's the issue mentioned above). For example:
```
t_0: Group 0                            t_1: Group 1
     0000000000000000 1000000000000000       0000000000000001 0000000000000001
```

It's obvious in this example that `t_0` is followed by `t_1`, but the conventional intersection would fail in this case. So to solve this I decided to do the intersection in two passes, the first calculates "normal" intersection and the second this annoying edge case.

**Note:** I don't know how Doug solved this, I haven't checked the code. But this issue is mentioned in the article.

# Use your index time wisely
In the field of Information Retrieval and Databases one way to reduce the search/query time is to pre calculate more during indexing/data ingestion.

One of the techniques that I used very early on in the making of this project is merging tokens during indexing (similar to [n-grams](https://en.wikipedia.org/wiki/N-gram)).

I had a few constraints when implementing this final index size, memory consumption while indexing and indexing time, I wanted to minimize the impact of this feature.

**Why do I want to minimize the impact ?** For the most time I developed this on a 16GB machine with a few hundred gigabytes left, so I was very constrained in this sense. And for indexing time sice I'm developing I want to iterate fast, so if I need to re-index the whole thing it can't take a long time.

**Note:** If you look at the source code on Github you will see that my indexing to this day is done on a single thread, the reason is that I can easily achive a high memory consuption on a sinle thread. The reason why it consumes so much memory is that most of the indexing is done and cached on RAM to be as fast as possible, indexing this 3.2M documents only take around 30/35m on a single thread.

## How to solve this problem ?
The idea is to only merge common tokens, and you might ask: "what is a common token ?" Well it's simple they are the tokens that appear the most in the collection. You can specify how many of the top tokens to consider as common ones, or as a percentage. I arbitrarily chose the top 50 tokens. There is also a parameter of the maximum sequence length, in this case I used 3.

Increasing this two parameters will make the index size, index memory consumption and indexing time grow, so it's a careful balance. But the more you compute at indexing time the better, if you can afford more go for it.

The good thing about merging common tokens is that they are the most expensive in general to compute the intersection, so removing them makes things a lot faster.

Here is and example, where `C_n` is a common token and `R_n` is a rare token (rare tokens are all of the other tokens that are not common).

`C_0 R_1 C_2 C_3 C_4 R_5 R_6 ... R_x C_y R_z ...`

This sequece of tokens will generate the following tokens and positions:
```
C_0: 0
C_0 R_1: 0
R_1: 1
R_1 C_2: 1
R_1 C_2 C_3: 1
C_2: 2
C_2 C_3: 2
C_2 C_3 C_4: 2
C_3: 3
C_3 C_4: 3
C_3 C_4 R_5: 3
C_4: 4
C_4 R_5: 4
R_5: 5
R_6: 6
...
R_x: x
R_x C_y: x
C_y: y
C_y R_z: y
R_z: z
```

Doing this allows us to reduce the number of intersections done at search time. 

**Why are you merging up to one rare token at the begining or at the end ?** Let's consider that someone searched for `C_0 R_1 C_2 C_3`. If we don't do this merge we would end up searching for `C_0`, `R_1`, `C_2 C_3` and this is bad, as established intersecting common tokens is a problem so it's way better to search `C_0 R_1`, `C_2 C_3`. I learned this the hard way...

This brings us to the next topic that is done only at search time, the **minimization** step.

# Dinamyc Programming in the wild
Let's use the same example as above, but this time the person searched for `R_1 C_2 C_3 C_4 R_5`. Since we have all possible combinations from the merge phase we can be smart and try to predict which combination of this tokens will take less time to be intersected. 

At search time we can be greedy while merging, but this might not lead to the fastest intersection combination of tokens. In the greedy version we will compute the intersection of `R_1 C_2 C_3`, `C_4 R_5`, but it might be better to compute `R_1`, `C_2 C_3 C_4`, `R_5` or `R_1 C_2`, `C_3 C_4 R_5` and so on...

It's 100% worth spending time here before computing the intersection, I learned this the hard way...

Does this look like some kind of problem to you ? Yes **Dinamyc Programming**, sometimes this problems appear in the wild, so yes Leet Code is not a lie (I don't like Leet Code).

How can we solve this ? First let's list what we need to do:
* List all possible combinations of tokens (in a smart way)
* Estimate the cost of the intersection for that combination (in a smart way).

Yeah this is expensive... Usually when you find a DP problem in the wild what do you do ? Google a solution for it, in my case this wasn't a possibility, I have created this problem, so now I need to solve it.

It didn't take too long to for me to get a solution, since the algo isn't that hard. We can also di use some memoization to amortize the cost. Here is the POC I wrote in python (very porly optimized) while trying to find a solution for this problem.

```python
# arr: tokens
# scores: score for each possible token combination
# N: maximum sequence len
def minimize(arr, scores, N):
    if len(arr) == 0:
        return (0, [])

    final_score = float('inf')
    choices = []
    e = min(N, len(arr))
    sub_arr = arr[:e]
    for j in range(len(sub_arr), 0, -1):
        sub_sub_arr = sub_arr[:j]
        rem = arr[j:]

        concated = ' '.join(sub_sub_arr)
        score = scores[concated]

        (rem_score, rem_choices) = minimize(rem, scores, N)
        calc_score = score + rem_score
        if calc_score < final_score:
            choices.clear()
            final_score = calc_score
            choices.append(concated)
            for r_choise in rem_choices:
                choices.append(r_choise)

    return (final_score, choices)
```

This version is very simple, it doesn't do the correct merging of tokens nor has any optimization/memoization, but what is important is the idea. As said previously we want to mimize the cost/score, since intersection is computed in `O(n+m)` that's what we are aiming to minimize, there is another approches to the cost function like trying to find the combination that leads to the smallest possible of a single token.

Let's do an example on how this works, since in here we just merge tokens without caring if they are common or rare, let's assume the following query `t_0 t_1 t_2 t_3 t_4`:

For each call of the function we loop over all possible merges of the remaining tokens. So we get the cost for the token `t_0 t_1 t_2` and call the function recursively for `t_3 t_4` and so on. The call graph would look something like this

```

minimize(t_0 t_1 t_2 t_3 t_4)
```