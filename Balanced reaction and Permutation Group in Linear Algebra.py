{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "d1fbdb3f-2d58-4857-83fc-c95b290863fb",
   "metadata": {},
   "source": [
    "---\n",
    "title: \"Balancing a Chemical Reaction in Linear Algebra\"\n",
    "date: \"2026-2-1\"\n",
    "categories: [Python 3, Jupyter Notebook, Numpy, Matplotlib, Caycley Table]\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "725febf2-0616-4bfc-bee3-0ccaa1a31892",
   "metadata": {},
   "source": [
    "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]("
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "38beae9f-1c93-4bdd-8d29-8845a8d56d2e",
   "metadata": {},
   "source": [
    "# Balancing a Chemical Reaction and Permutation Group in Linear Algebra\n",
    "\n",
    "## Linear Algebra\n",
    "Linear Algebra is the branch of mathematics that focuses on the study of vectors, vector spaces, matrices, and linear tranformations. It deals with linear equations, linear functions, and their representations through matrices and determinants. It has a wide range of appliation in Physics and Mathematics, It is the basic concept for machine learning and data science.\n",
    "\n",
    "## Foundations of Linear \n",
    "Linear Algebra is trikingly similar to the algebra we learned in hihg school, except that in the place of ordinary single numbers, it deals with vectors. Elementary linear algebra introduces the foundational concepts that form the building blocks of the subject. It covers basic operations on matrices, solving system of equations, and understanding vectors.\n",
    "1. Scalar\n",
    "2. Vector\n",
    "3. Vector Spaces\n",
    "4. Matrices\n",
    "5. Matrix Operations\n",
    "\n",
    "## 1. Scalar\n",
    "A scalar is a number. Examples of scalars are temperature, distances, speed, or mass, all quantities that have a magnitude but no \"direction\" other than perhaps positive or negative.\n",
    "\n",
    "## 2. Vectors\n",
    "In mathematics, vectors are fundamental objects that represent quantities with both magnitude and direction. A vector is _a list of numbers_. There are (at least) two ways to intrepret what this list of numbers mean: One way to think of the vector as being _a point in a space_. Another way to think of a vector is _a magnitude and a direction_.\n",
    "\n",
    "## 3. Vector spaces\n",
    "A basis set is  a linearly independent set of vectors that, when used in linear combinations, can combination represent every vector in agiven vector space. All vectors live within a _vector space_. A vector space is exactly what it sounds like the space in which vectors live. The vector space is intuitively spatial since all available directions of motion can be plotted directly onto a spacial map of the room.\n",
    "\n",
    "## 4. Matrices\n",
    "A matrix, like a vector, is also a collection of numbers. The difference is that a matrix is a set of numbers rather than a list. Many of the same rules we juct outlined for vectors above apply equally well to matrices. Matrices are rectangular arrays of numbers, symbols, or characters where all of these elements are arranged in each row and column. \n",
    "- A matrix is identified by its order, which is given in the form of rows x and columns and the location of each element is given by the row and column it belongs to.\n",
    "- A matrix is represented as ($[P]_{m \\times n}$), where P is the matrix, m is the number of rows and n is the number of columns.\n",
    "\n",
    "\n",
    "## 5. Matrix operations\n",
    "Matrix operations mainly involve three algebraic operations, which are the addition of matrices, subtraction of matrices, and multipication of matrices. To add or substract matrices, these must be of identical order, and for multipication, the number of columns in the first matrix equals the number of rows in the second matrix. Examples of matrix operations:\n",
    "- transpose, sum and difference, scalar multipication.\n",
    "- matrix multipication, matrix-vector product.\n",
    "- matrix inverse.\n",
    "\n",
    "\n",
    "# Linear Algebra in Chemistry\n",
    "Linear Algebra is the study of vectors in vector spaces, and linear transformation in that vector spaces. The study of vector spaces and linear transformations in said spaces is called Linear Algebra. In this work, we will solving a set of linear equation for balancing reaction a chemical reaction.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9864ef86-dd6f-4f2c-92d7-dadfd6e6cfdf",
   "metadata": {},
   "source": [
    "## P18.1 Exercise: Balancing a redox reaction\n",
    "Balance the equation ofr the reaction of permanganate and iodide ions in basic solution:\n",
    "<p align='center'>\n",
    "    $MnO_{4}^{-} (aq) + I^{-} (aq) \\longrightarrow MnO_{2} (s) + I_{2} (aq)$\n",
    "</p>\n",
    "\n",
    "__Hint__: You will need to add $OH^{-}$ ions $H_{2}O$ molecules to the reaction in stoichiometric amounts to be determined.\n",
    "\n",
    "In this redox reaction, the hydroxide ions and water molecules added before it can be balanced:\n",
    "<p align='center'>\n",
    "    $aMnO_{4}^{-} (aq) + bI^{-} (aq) + cH_{2}O (l) \\longrightarrow dMnO_{2} (s) + eI_{2} (aq) + fOH^{-} (aq)$\n",
    "</p>\n",
    "We must also conserve the charge, which leads to the following four equations:\n",
    "<p align='center'>\n",
    "    \n",
    "    Mn: a + d = 0\n",
    "    O:  4a + c + 2d + f = 0\n",
    "    I:  b + 2e = 0\n",
    "    H:  2c + f = 0\n",
    "    charge: a + b + f = 0\n",
    "</p>\n",
    "\n",
    "The problem is undertermined but we can set the constraint a = 1 since we know we can scale all the coefficients by the same amount and keep the reaction balanced. Therefore, in matrix form:\n",
    "<p align='center'>\n",
    "    $\\begin{pmatrix} 1 & 0 & 0 & 1 & 0 & 0\\\\4 & 0 & 1 & 2 & 0 & 1\\\\0 & 1 & 0 & 0 & 2 & 0\\\\0 & 0 & 2 & 0 & 0 & 1\\\\1 & 1 & 0 & 0 & 0 & 1\\\\1 & 0 & 0 & 0 & 0 & 0 \\end{pmatrix}  \\begin{pmatrix} a\\\\b\\\\c\\\\d\\\\e\\\\f  \\end{pmatrix} = \\begin{pmatrix} 0\\\\0\\\\0\\\\0\\\\0\\\\1 \\end{pmatrix}$\n",
    "</p>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "0d6f563d-879e-4a8c-aa95-5812261f33c5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "a, b, c, d, e, f = [ 1.   3.   2.  -1.  -1.5 -4. ]\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "A = np.array([[1, 0, 0, 1, 0, 0], # Mn\n",
    "              [4, 0, 1, 2, 0, 1], # O\n",
    "              [0, 1, 0, 0, 2, 0], # I\n",
    "              [0, 0, 2, 0, 0, 1], # H\n",
    "              [1, 1, 0, 0, 0, 1], # charge\n",
    "              [1, 0, 0, 0, 0, 0]]) # constraint a = 1\n",
    "x = np.array([0, 0, 0, 0, 0, 1])\n",
    "coeffs = np.linalg.solve(A, x)\n",
    "print('a, b, c, d, e, f =', coeffs)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9789d6f1-614c-4381-8b4b-1c982144ddeb",
   "metadata": {},
   "source": [
    "The balanced reaction is therefore:\n",
    "<p align='center'>\n",
    "    $2\\;MnO_{4}^{-} (aq) + 6\\;I^{-} (aq) + 4\\;H_{2}O (l) \\longrightarrow 2\\;MnO_{2} (s) + 3\\;I_{2} (aq) + 8\\;OH^{-} (aq)$\n",
    "</p>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a4d0ec96-2240-4eff-9afe-07352d9a8e8e",
   "metadata": {},
   "source": [
    "## P18.2 Exercise: Balancing a complex reaction\n",
    "Balance the following equation [R. J. Stout, _J.Chem.Educ.__72__, 1125 (1995)]: \n",
    "<p align='center'>\n",
    "    $aCr_{7}N_{66}H_{96}C_{42}O_{24} + MnO_{4}^{-} + H^{+} \\longrightarrow Cr_{2}O_{7}^{2-} + Mn^{2+} + CO_{2} + NO_{3}^{-} + H_{2}O$\n",
    "</p>\n",
    "The stoichiometric constraints can be written as a sequence of three equations, one for each of the atoms Cr, N, H, C, O, and Mn.\n",
    "<p align='center'>\n",
    "\n",
    "        Cr: 7a + 2d = 0\n",
    "        N : 66a + g = 0\n",
    "        H : 96a + c + 2h = 0\n",
    "        C : 42a + f = 0\n",
    "        O : 24a + 4b + 7d + 2f + 3g + h = 0\n",
    "        Mn: b + e = 0\n",
    "    Charge: b + c + d + e + g = 0\n",
    "</p>\n",
    "Again, The problem is undertermined but we can set the constraint a = 1 since we know we can scale all the coefficients by the same amount and keep the reaction balanced. Therefore, in matrix form:\n",
    "<p align='center'>\n",
    "    $\\begin{pmatrix} 7 & 0 & 0 & 2 & 0 & 0 & 0 & 0\\\\ 66 & 0 & 0 & 0 & 0 & 0 & 1 & 0\\\\ 96 & 0 & 1 & 0 & 0 & 0 & 0 & 2\\\\ 42 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\\\ 24 & 4 & 0 & 7 & 0 & 2 & 3 & 1\\\\ 0 & 1 & 0 & 0 & 1 & 0 & 0 & 0\\\\ 0 & -1 & 1 & -2 & 2 & 0 & -1 & 0\\\\ 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\end{pmatrix} \\begin{pmatrix} a\\\\b\\\\c\\\\d\\\\e\\\\f\\\\g\\\\h \\end{pmatrix} = \\begin{pmatrix} 0\\\\0\\\\0\\\\0\\\\0\\\\0\\\\0\\\\1 \\end{pmatrix}$\n",
    "</p>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "8f0e403f-be1c-452e-8412-3b3fa7f23e78",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "a, b, c, d, e, f, g, h= [   1.   117.6  279.8   -3.5 -117.6  -42.   -66.  -187.9]\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "A = np.array([[7, 0, 0, 2, 0, 0, 0, 0], \n",
    "              [66, 0, 0, 0, 0,0, 1, 0], \n",
    "              [96, 0, 1, 0, 0, 0, 0, 2], \n",
    "              [42, 0, 0, 0, 0, 1, 0, 0], \n",
    "              [24, 4, 0, 7, 0, 2, 3, 1], \n",
    "              [0, 1, 0, 0, 1, 0, 0, 0], \n",
    "              [0, -1, 1, -2, 2, 0, -1, 0], \n",
    "              [1, 0, 0, 0, 0, 0, 0, 0]])\n",
    "\n",
    "x = np.array([0, 0, 0, 0, 0, 0, 0, 1])\n",
    "coeffs = np.linalg.solve(A, x)\n",
    "print('a, b, c, d, e, f, g, h=', coeffs)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "876d87db-30ae-4d29-8216-2d91a5702cdd",
   "metadata": {},
   "source": [
    "The balanced reaction is therefore:\n",
    "<p align='center'>\n",
    "    $10\\;Cr_{7}N_{66}H_{96}C_{42}O_{24} + 1176\\;MnO_{4}^{-} + 2798\\;H^{+} \\longrightarrow 35\\;Cr_{2}O_{7}^{2-} + 11760\\;Mn^{2+} + 420\\;CO_{2} + 660\\;NO_{3}^{-} + 1879\\;H_{2}O$\n",
    "</p>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0c751d68-c363-4c36-b2de-29e2c9212177",
   "metadata": {},
   "source": [
    "## P18.3 Exercise: The reaction between copper and nitric acid\n",
    "Copper metal may be thought of as reacting with nitric acid according to the followeing reaction:\n",
    "<p align='center'>\n",
    "    $aCu\\;(s) + bHNO_{3}\\;(aq) \\longrightarrow cCu(NO_{3})_{2}\\;(aq) + dNO\\;(g) + eNO_{2}\\;(g) + fH_{2}O\\;(l)$\n",
    "</p>\n",
    "Show that this reaction is carried put with concentrated nitric acid the favored gaseous product is nitrogen dioxide instead of nitric oxide (d = 0); conversely, in dilute nitric acid nitric oxide is produced of $NO_{2}$ (e = 0). Write balanced equations for these two cases.\n",
    "\n",
    "\n",
    "With concentrate $HNO_{3}$, we can eliminate the column corresponding to NO since $NO_{2}$ is observed to be the product.\n",
    "<p align='center'>\n",
    "    $aCu\\;(s) + bHNO_{3}\\;(aq) \\longrightarrow cCu(NO_{3})_{2}\\;(aq) + dNO_{2}\\;(g) + eH_{2}O\\;(l)$\n",
    "</p>\n",
    "\n",
    "The equations giverning the balance of the reaction can be written as,\n",
    "<p align='center'>\n",
    "\n",
    "    Cu: a + c = 0\n",
    "    H:  b + 2e = 0\n",
    "    N:  b + 2c + d = 0\n",
    "    O: 3b + 6c + 2d + e = 0\n",
    "</p>\n",
    "Again, The problem is undertermined but we can set the constraint a = 1 since we know we can scale all the coefficients by the same amount and keep the reaction balanced. Therefore, in matrix form:\n",
    "<p align='center'>\n",
    "    $\\begin{pmatrix} 1 & 0 & 1 & 0 & 0\\\\ 0 & 1 & 0 & 0 & 2\\\\ 0 & 1 & 2 & 1 & 0\\\\ 0 & 3 & 6 & 2 & 1\\\\1 & 0 & 0 & 0 & 0\\end{pmatrix} \\begin{pmatrix} a\\\\b\\\\c\\\\d\\\\e \\end{pmatrix} = \\begin{pmatrix} 0\\\\0\\\\0\\\\0\\\\1 \\end{pmatrix}$\n",
    "</p>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "4e6efb72-8836-45c7-8a7f-28affd773173",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "a, b, c, d, e= [ 1.  4. -1. -2. -2.]\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "A = np.array([[1, 0, 1, 0, 0],\n",
    "              [0, 1, 0, 0, 2],\n",
    "              [0, 1, 2, 1, 0],\n",
    "              [0, 3, 6, 2, 1],\n",
    "              [1, 0, 0, 0, 0]])\n",
    "\n",
    "x = np.array([0, 0, 0, 0, 1])\n",
    "coeffs = np.linalg.solve(A, x)\n",
    "print('a, b, c, d, e=', coeffs)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "87d77919-0707-4cde-9b4a-0fc14d8331bc",
   "metadata": {},
   "source": [
    "The balanced reaction in this case is therefore:\n",
    "<p lign='center'>\n",
    "    $Cu\\;(s) + 4\\;HNO_{3}\\;(aq) \\longrightarrow Cu(NO_{3})_{2}\\;(aq) + 2\\;NO_{2}\\;(g) + 2\\;H_{2}O\\;(l)$\n",
    "</p>\n",
    "\n",
    "In dilute nitric acid, we can eliminate the column corresponding to $NO_{2}$ instead:\n",
    "<p align='center'>\n",
    "    $aCu\\;(s) + bHNO_{3}\\;(aq) \\longrightarrow cCu(NO_{3})_{2}\\;(aq) + dNO\\;(g) + eH_{2}O\\;(l)$\n",
    "</p>\n",
    "\n",
    "The equations giverning the balance of the reaction can be written as,\n",
    "<p align='center'>\n",
    "\n",
    "    Cu: a + c = 0\n",
    "    H:  b + 2e = 0\n",
    "    N:  b + 2c + d = 0\n",
    "    O: 3b + 6c + d + e = 0\n",
    "</p>\n",
    "Again, The problem is undertermined but we can set the constraint a = 1 since we know we can scale all the coefficients by the same amount and keep the reaction balanced. Therefore, in matrix form:\n",
    "<p align='center'>\n",
    "    $\\begin{pmatrix} 1 & 0 & 1 & 0 & 0\\\\ 0 & 1 & 0 & 0 & 2\\\\ 0 & 1 & 2 & 1 & 0\\\\ 0 & 3 & 6 & 1 & 1\\\\1 & 0 & 0 & 0 & 0\\end{pmatrix} \\begin{pmatrix} a\\\\b\\\\c\\\\d\\\\e \\end{pmatrix} = \\begin{pmatrix} 0\\\\0\\\\0\\\\0\\\\1 \\end{pmatrix}$\n",
    "</p>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "19c24c3b-c428-43d0-91fa-1aaaada719c1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "a, b, c, d, e= [ 1.          2.66666667 -1.         -0.66666667 -1.33333333]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([ 3.,  8., -3., -2., -4.])"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import numpy as np\n",
    "A = np.array([[1, 0, 1, 0, 0],\n",
    "              [0, 1, 0, 0, 2],\n",
    "              [0, 1, 2, 1, 0],\n",
    "              [0, 3, 6, 1, 1],\n",
    "              [1, 0, 0, 0, 0]])\n",
    "\n",
    "x = np.array([0, 0, 0, 0, 1])\n",
    "coeffs = np.linalg.solve(A, x)\n",
    "print('a, b, c, d, e=', coeffs)\n",
    "coeffs * 3 # To get integer coefficients, multiply by 3"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "53d107e3-db57-429a-a73e-476a92ef33b7",
   "metadata": {},
   "source": [
    "The balanced reaction in this case is\n",
    "<p lign='center'>\n",
    "    $3Cu\\;(s) + 8\\;HNO_{3}\\;(aq) \\longrightarrow 3Cu(NO_{3})_{2}\\;(aq) + 2\\;NO\\;(g) + 4\\;H_{2}O\\;(l)$\n",
    "</p>\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bcbea323-0d64-45a4-bbdc-5d987b236c14",
   "metadata": {},
   "source": [
    "## P18.4 Construction the Cayley Table for a Permutation Group\n",
    "What is the most effective way to shuffle a pack of cards?. To answer this question we need to return to the ideas about permutaions. We will see that permutations provide an example of a mathematical structure called a _groups._ Groups plan an important role in many areas of mathematics.\n",
    "\n",
    "Caycle's theorem states that every group G is isomorphic to a subgroup of the symetric group acting on G. In this context it means that if we have a vector of permutations tha comprise a group, then we can nicely represent its stucture using a table. \n",
    "\n",
    "The distinct rearrangements of three objects, [1, 2, 3], can be described by six distinct permutations, written in cycle notation as \n",
    "<p align='center'>\n",
    "    e, (12), (13), (23), (123), (132).\n",
    "</p>\n",
    "\n",
    "Here, e represent the identity permutation (which leaves the objects undisturbed) and, for example, (12) swaps objects 1 and 2; (123) is the cyclic permutation ($1\\rightarrow 2, 2\\rightarrow 3, 3\\rightarrow 1$).\n",
    "\n",
    "- (a) Demonstrate that the inverse of each $3\\times3$ permuation matrix is its transpose (i.e., the permutation matrices are _orthogonal_)\n",
    "- (b) Show that there is some power of each permutation matrix that is the identity matrix.\n",
    "- (c) The _parity_ of a permutation is -1 or +1 according to whether the number of distinct pairwise swaps it can be broken down into is odd or even, repectively. Show that the parity of each permutation is equal to the determinant of its corresponding permutation matrix.\n",
    "\n",
    "## Mathematical meaning\n",
    "This code is finding the order of each element in the group $S_{3}$:\n",
    "- The order of an element g in a group is the smallest positive integer n such that $g^{n}=e$.\n",
    "- If true, it prints that the matrix is __orthogonal__ (since orthogonal matrices satisfy $m^{-1}=m^{T}$).\n",
    "- In $S_{3}$:\n",
    "  - The identity e has order 1\n",
    "  - Transpositions like (12), (13), (23)have order 2.\n",
    "  - 3-cycles like (123), (132) have order 3.\n",
    "\n",
    "## The key arguments:\n",
    "- `np.eye(3, dtype=int)`: creates the 3 x 3 identity matrix (the permutation 'e').\n",
    "- `np.array([...], dtype=int)`: the nested list `[[...], [...], [...]]` species the matrix.\n",
    "- `np.allclose(minv, m.T)`: compares two matrices element-wise to check if they are approximately equal.\n",
    "- `minv`: the inverse of the matrix.\n",
    "- `m.T`: the transpose of the matrix.\n",
    "- `enumerate`: `s` (list of matrices), iterates through the list `s`, returning both index `i` and matrix `m`.\n",
    "- `np.linalg.matrix_power(m, n)`: `m` (matrix), `n` (integer)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "b5c92cc3-f930-443a-801d-4104a0b7a997",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "s[0] is orthogonal\n",
      "s[1] is orthogonal\n",
      "s[2] is orthogonal\n",
      "s[3] is orthogonal\n",
      "s[4] is orthogonal\n",
      "s[5] is orthogonal\n",
      "s[1]^2 = e\n",
      "s[2]^2 = e\n",
      "s[3]^2 = e\n",
      "s[4]^2 = e\n",
      "s[4]^3 = e\n",
      "s[5]^2 = e\n",
      "s[5]^3 = e\n",
      "    e:  1\n",
      " (12): -1\n",
      " (13): -1\n",
      " (23): -1\n",
      "(123):  1\n",
      "(132):  1\n"
     ]
    }
   ],
   "source": [
    "# (a) Demonstrate that the inverse of each 3 x 3 permuation matrix is its transpose (i.e., the permutation matrices are _orthogonal_)\n",
    "\n",
    "import numpy as np\n",
    "s = [None] * 6\n",
    "names = ['e', '(12)', '(13)', '(23)', '(123)', '(132)']\n",
    "s[0] = np.eye(3, dtype=int)\n",
    "s[1] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=int)\n",
    "s[2] = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=int)\n",
    "s[3] = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=int)\n",
    "s[4] = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=int)\n",
    "s[5] = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=int)\n",
    "\n",
    "for i, m in enumerate(s):\n",
    "    minv = np.linalg.inv(m)\n",
    "    if np.allclose(minv, m.T):\n",
    "        print(f's[{i}] is orthogonal')\n",
    "\n",
    "# (b) Show that there is some power of each permutation matrix that is the identity matrix.\n",
    "for i, m in enumerate(s):\n",
    "    n = 1\n",
    "    while not np.allclose(np.linalg.matrix_power(m, n), s[0]):\n",
    "        n += 1\n",
    "        print(f's[{i}]^{n} = e')\n",
    "\n",
    "# c) The _parity_ of a permutation is -1 or +1 according to whether the number of distinct pairwise swaps it can be broken down into is odd or even, repectively.\n",
    "parities = [1, -1, -1, -1, 1, 1]\n",
    "for name, parity in zip(names, parities):\n",
    "    print(f'{name:>5s}: {parity:2d}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cf50b92d-e321-408c-be14-b485a884012d",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.14.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
