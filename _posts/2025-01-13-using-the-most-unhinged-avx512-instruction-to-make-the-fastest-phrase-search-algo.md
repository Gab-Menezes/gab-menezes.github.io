---
layout: post
title: "Using the most unhinged AVX512 instruction to make the fastest phrase search algo"
---

# Disclaimers before we start
* The contents of this blog post are inspired by the wonderful idea of [Doug Turbull](https://softwaredoug.com/) from the series of blog posts about [Roaringish](https://softwaredoug.com/blog/2024/01/21/search-array-phrase-algorithm). In here we will take this ideas to an extreme, from smart algos to raw performance optimization.
* I highly recommend reading the [Roaringish](https://softwaredoug.com/blog/2024/01/21/search-array-phrase-algorithm) blog post, but if you don't want there will be a recap on how it works.
* This project it's been almost 7 months in the making. At the beginning I wasn't planning on writing a blog post about it, so that's why the structure of this post will be a little bit different. I will go through all major changes in my git history explaning how and why this impacted the performance (sometimes we will go deep and others will be a brief overview). And since there is a lot of things to go through (and I wasn't planning on writing this...) there will be no benchmarks on this older changes, but I can ensure you everything was throughly benchmarked, before being commited to the repo. (This will be a long blog post...)
* **We won't talk about**:
  * Multi threading, since this workload is trivially parallelizable (by sharding) it's kinda a easy optimization that anyone can do.
  * Things outside my code to make it faster, like enabling Huge Pages.
  * The main focus will be in the search part, so there will be no focus in the indexing part, only when needed
* **Benchmark methodology**:
  * We are optimizing for raw latency, the lower the better.
  * I have randomly select a bunch of queries (each with different bottleneck profile) and measured how each change impacted it's performance.
  * I ran each query 20 times to warmup the CPU, after that I ran the same query another 1000 times and collected the time taken as a whole and in each crucial step, with this we have time/iter.
  * For those who say that that's not a good benchmark and I should have used criterion with statistical analysis and blah blah blah... All I can say is both of my system (spoiler alert) are pretty reproducible (up to 5 us on the whole benchmark, not per iter, per benchmark run !!!) if I run the same benchmark twice. So this is good enough.
  * So after the CPU is warm the time in each iteration is pretty damn consistent, so that's why I consider this good enough.
  * The dataset used is the same used by Doug, [MS MARCO](https://microsoft.github.io/msmarco/), containing 3.2M documents, around 22GB of data. It consists of a link, question and a long answer, in this case we only index the answer (so 20/22GB of data). 
  * Getting close to the end there will be some benchmarks and to finalize there will be a comparision against [Meilisearch](https://www.meilisearch.com/) (production ready Search Engine, who is known for it's good performance)
* Also a huge thanks to all of the people who helped me through this, specially the super kind people on the [Gamozo's discord](https://discord.gg/gwfvzzQC), how through the last year had to see me go crazy about intersection algos.
* **Why am I doing this ?** Because I like it and wanted to nerd snipe some people.

# Where it all began
Around 7 months ago I found the wonderful blog post about [Roaringish](https://softwaredoug.com/blog/2024/01/21/search-array-phrase-algorithm) and after reading it and falling in love with the idea I decided to implement my own version in Rust, since the original was implemented in Python. At the beginning I wasn't planning on investing so much time (even though I did this in my free time), but I got so invested in the idea that decided to keep pushing and pushing until a fixed deadline that I stipulated myself (because if not I would be still writing code).

## What are we doing ? And why we care ?
Do you know when you go to your favorite search engine and search for something using double quotes, like passage of a book/article or something very specific ? That's called phrase search (sometimes exact search), what we are telling the search engine is that we want this exact words in this exact order (this varies from search engine to search engine, but that's the main idea). In contrast when searching by keywords (not in double quotes) we don't care about the order or if it's the exact word, it may be a variation.

There is one big difference between this two, searching by keywords is relative cheap when compared to doing a phrase search. In both cases we are calculating the intersection between the reserve indexes (in the keyword search we may take the union, but for the sake of simplicity let's assume the intersection), but in the phrase search we need to keep track where each token appeared to guarantee that we are looking for tokens close to each other.

So this makes phrase search way more computationally expensive, the convetional algo found in books is very slow, let's take a look at this first and compare with Doug's brilliant idea.

### How the conventional algo works
![](/assets/2025-01-13-using-the-most-unhinged-avx512-instruction-to-make-the-fastest-phrase-search-algo/convetional-phrase-search-algo.png)

This algo (taken from Information Retrieval: Implementing and Evaluating Search Engines) analyzes one document at the time, so the `nextPhrase` function needs to called for each document in the intersection of document ids in the index.

Description (also taken from the book): Locates the first occurrence of a phrase after a given position. The function call the `next(t, i)` method returns next position `t` after the position i. Similar to `prev(t, i)`.

You don't need to understand this algo, just that is inefficient because of a lot of reasons:
  * Intersection is expensive
  * Analyzes one document and the time
  * Not cache friendly we are jumping around with the `next` and `prev` functions
  * Recursive

For a small collection of documents this works fine, but imagine for large collections 1M+ documents this will blow up quiclky (hundreds of milliseconds, maybe even seconds per query), which is unacceptable for a real time search engine, so not good at all.


### The genius idea
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

To do a phrase search we can connect two terms at the time, the left and right one, previoud one, i.g searching by "mary had a little lamb", will result in searching by:

* `"mary"` and `"had"` = `"mary had"`
* `"mary had"` and `"a"` = `"mary had a"`
* ...

So we reuse the work done the previous step, just connect the right term with the previous. Imagine the scenario where `"mary had"` occurs in the following positions: `[1, 6]` and `"a"` appers in the position `[2]`, so `"mary had a"` occurs in the positions `[2]`, we keep doing this for the next token until we finish it.

The main idea to recreate and optimize this behaviour was taken from [`Roaring Bitmaps`](https://roaringbitmap.org/), that's why it's called `Roaringish`. We want to pack as much data as possible and avoid storing the positions for each document for each term separatly.

Assuming that the $$pos \le 2^{16}*16 = 1048576$$ (i.e the maximum document length is 1048576 tokens which is very reasonable) we can decompose this value into two 16 bits, one representing the group and other the value, where $$pos = group * 16 + value$$

```python
posns  =              [1, 5, 20, 21, 31, 100, 340]
groups = posns // 16 #[0, 0,  1,  1,  1,   6,  21]
values = posns %  16 #[1, 5,  4,  5, 15,   4,   4]
```

Since $$value \lt 15$$, we can pack even more data by bitwising each group value into single 16 bit number

```
Group 0             Group 1            ...   Group 21
0000000000100010    1000000000110000   ...   0000000000010000
(bits 1 and 5 set)  (bits 4, 5, 15 set)      
```

With this we can pack the group and the values in a single 32 bit number, by shifting the group and or it with the packed values i.e $$group\_value = (group << 16) \mid values$$.

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

With this "little lamb" is found in group 21 value 3, i.e $$21*16+3 = 339$$.

And the magical part is: To avoid having to analyze one document at the time we can pack the document id (assuming 32 bit document id) into the 32 MSB of a 64 bit number, while the LSB are the group and value. $$packed = (doc\_id << 32) \mid group\_value$$

When calculating the intersection we take the document id and group (48 MSB's), with this we can search the whole index in a single shot. So in the end we have a single continuous array of data, with all document ids and positions that contain that token.

Pretty f* cool.

There are some annoying issues that we will deal with them in the future:
* When the values bits are in the boundry of the groups
```
Group 0                            Group 1
0000000000000000 1000000000000000  0000000000000001 0000000000000001
```
* The maximum distance between tokens is 1 at the moment, that's why we shift left by 1. But there is the concept of `slop` where you want the token: $$t_0$$ to be a distance $$N$$ of token $$t_1$$.

# Where my journey began
After reading this genius idea I had to implement this in Rust, specially because the original version was implemented in Python, so of course there was performance being left on the table.