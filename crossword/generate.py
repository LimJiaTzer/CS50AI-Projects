import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """

        for variable, words in self.domains.items():
                possible_words = set()
                for word in words:
                    if len(word) == variable.length:
                        possible_words.add(word)
                self.domains[variable] = possible_words

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        overlap = self.crossword.overlaps.get((x, y))
        revision = False
        if overlap:
            i, j = overlap
            possible_words = set()
            for word1 in self.domains[x]:
                if any(word1[i] == word2[j] for word2 in self.domains[y]):
                    possible_words.add(word1)
            if len(self.domains[x]) != len(possible_words):
                revision = True
                self.domains[x] = possible_words
        return revision

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        queue = []

        if arcs is not None:
            queue.extend(arcs)
        else:
            for variable in self.domains:
                for neighbour in self.crossword.neighbors(variable):
                    queue.append((variable, neighbour))
        while queue:
            x, y = queue.pop(0)
            if self.revise(x, y):
                if not self.domains[x]:
                    return False
                for neighbour in self.crossword.neighbors(x):
                    if neighbour != y:
                        queue.append((neighbour, x))
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        return set(assignment.keys())==self.crossword.variables

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        words_used = set()
        for variable, word in assignment.items():
            check_length = (variable.length == len(word))
            check_distinct = word not in words_used
            words_used.add(word)
            check_arc = True
            for neighbour in self.crossword.neighbors(variable):
                if neighbour in assignment:
                    neighbour_word = assignment[neighbour]
                    overlap = self.crossword.overlaps.get((variable, neighbour))
                    if overlap:
                        i, j = overlap
                        if word[i] != neighbour_word[j]:
                            check_arc = False
                            break

            if check_length and check_distinct and check_arc:
                continue
            else:
                return False
        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        neighbours = self.crossword.neighbors(var)
        list_of_word = []

        for word in self.domains[var]:
            n = 0
            for neighbour in neighbours:
                if neighbour not in assignment:
                    overlap = self.crossword.overlaps.get((var, neighbour))
                    if overlap:
                        i , j = overlap
                        for neigbour_word in self.domains[neighbour]:
                            if word[i] != neigbour_word[j]:
                                n+=1
            list_of_word.append((word, n))
        list_of_word.sort(key=lambda x:x[1])
        output = []
        for x in list_of_word:
            output.append(x[0])
        return output

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        list_of_unchosen = []
        for v in self.domains:
            if v not in assignment:
                list_of_unchosen.append(v)
        if not list_of_unchosen:
            return None

        return min(list_of_unchosen, key=lambda x:(len(self.domains[x]), -len(self.crossword.neighbors(x))))

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        if self.assignment_complete(assignment):
            return assignment

        var = self.select_unassigned_variable(assignment)
        if var is None:
            return None
        for word in self.order_domain_values(var, assignment):
            saved_domains = self.deepcopy(self.domains)
            assignment[var] = word
            if self.consistent(assignment):
                arcs_to_check = []
                for neighbour in self.crossword.neighbors(var):
                    if neighbour not in assignment:
                        arcs_to_check.append((var, neighbour))
                consistent_after_check = self.ac3(arcs_to_check)
                if consistent_after_check:
                    result = self.backtrack(assignment)
                    if result != None:
                        return result

            del assignment[var]
            self.domains = saved_domains
        return None

    def deepcopy(self, domains):
        """
        Function to make a deepcopy a domain to perform ac3 in backtracking
        """
        copy_of_domains = dict()
        for key, value in domains.items():
            copy_of_domains[key] = value.copy()
        return copy_of_domains


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
