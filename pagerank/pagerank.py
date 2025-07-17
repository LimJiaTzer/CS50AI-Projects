import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    probability_distribution = dict()
    no_of_next_page = len(corpus[page])
    if no_of_next_page == 0:
        return {page:1/len(corpus) for page in corpus}
    p_of_any_pages = (1 - damping_factor) / len(corpus)
    p_of_each_nextpage = (damping_factor / no_of_next_page) + p_of_any_pages
    
    for link in corpus:
        if link in corpus[page]:
            probability_distribution[link] = p_of_each_nextpage
        if link not in corpus[page]:
            probability_distribution[link] = p_of_any_pages
    return probability_distribution
    

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    PageRank = dict()
    all_pages = [page for page in corpus]
    starting_position = random.choices(all_pages)[0]
    PageRank[starting_position] = PageRank.get(starting_position, 0) + 1
    for i in range(1, n):
        pd = transition_model(corpus, starting_position, damping_factor)
        next_position = random.choices(list(pd.keys()), weights=list(pd.values()), k=1)[0]
        PageRank[next_position] = PageRank.get(next_position, 0) + 1
        starting_position = next_position

    for j in PageRank:
        PageRank[j] /= n
    return PageRank

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pagerank = dict()
    N = len(corpus)
    for page in corpus:
        pagerank[page] = float(1/N)
    
    def summation(pagerank, p):
        total = 0
        for x in corpus:
            if not corpus[x]:
                total += float(pagerank[x]/N)
            elif p in corpus[x]:
                total += float(pagerank[x]/len(corpus[x]))
        return total
    

    while True:
        new_pagerank = dict()
        d = float('-inf')

        for p in corpus:
            new_pagerank.setdefault(p,float((1-damping_factor)/N))
            new_pagerank[p] += damping_factor*summation(pagerank, p)

            d = max(abs(new_pagerank[p]-pagerank[p]), d)
        
        pagerank = new_pagerank
        if d<0.001:
            break
    return pagerank

if __name__ == "__main__":
    main()
