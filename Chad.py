import numpy as np
import copy

whiteCastle = {(2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4), (4, 2), (4, 3), (4, 4)}
blackCastle = {(7, 7), (7, 8), (7, 9), (8, 7), (8, 8), (8, 9), (9, 7), (9, 8), (9, 9)}
whiteWall = {(1,2),(1,3),(1,4),(5,2),(5,3),(5,4),(2,1),(3,1),(4,1),(2,5),(3,5),(4,5)}
blackWall = {(7,6),(8,6),(9,6),(7,10),(8,10),(9,10),(6,7),(6,8),(6,9),(10,7),(10,8),(10,9)}
walls = None

class Piece:
    def __init__(self, position, pieceColor, pieceType):
        self.pos = position
        self.color = pieceColor
        self.type = pieceType

    def search(self, deltas, board):
        moves = []
        for delta in deltas:
            moves.extend(self.lineSearch(delta, board))
        return moves

    def lineSearch(self, delta, board):
        col = self.pos[0] + delta[0]
        row = self.pos[1] + delta[1]
        res = []
        while 0 <= col < 12 and 0 <= row < 12 and board[col][row] is None:
            res.append((col, row))
            col = col + delta[0]
            row = row + delta[1]

        if 0 <= col < 12 and 0 <= row < 12 and self.canCapture(board[col][row]):
            res.append((col, row))

        return res

    def canCapture(self, other):
        if other is None or other.color == self.color:
            return False
        if other.type == 'k' or self.rightOfCapture(other) or other.rightOfCapture(self):
            return True

    def rightOfCapture(self, other):
        return self.inOwnCastle() and other.onOtherWall()

    def onOtherWall(self):
        return self.pos in whiteWall if self.color else self.pos in blackWall

    def inOwnCastle(self):
        return self.pointInOwnCastle(self.pos)

    def pointInOwnCastle(self, point):
        return point in blackCastle if self.color else point in whiteCastle

    def inOtherCastle(self):
        return self.pos in whiteCastle if self.color else self.pos in blackCastle


    def move(self, moveTo):
        self.pos = moveTo


class Rook(Piece):
    def __init__(self, position, pieceColor):
        super().__init__(position, pieceColor, 'r')

    def validMoves(self, board):
        return self.search([(1, 0), (-1, 0), (0, 1), (0, -1)], board)

    def netRep(self):
        return 1 if self.color else -1

    def __repr__(self):
        letter = "R" if self.color else "r"
        place = chr(ord('a') + self.pos[0]) + chr(ord('A') + self.pos[1])
        return letter + place


class Queen(Piece):
    def __init__(self, position, pieceColor):
        super().__init__(position, pieceColor, 'q')

    def validMoves(self, board):
        return self.search([(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)], board)

    def netRep(self):
        return 2 if self.color else -2

    def __repr__(self):
        letter = "Q" if self.color else "q"
        place = chr(ord('a') + self.pos[0]) + chr(ord('A') + self.pos[1])
        return letter + place


class King(Piece):
    def __init__(self, position, pieceColor):
        super().__init__(position, pieceColor, 'k')

    def validMoves(self, board):
        deltas = {(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2),
                  (1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1)}
        res = []
        for delta in deltas:
            col = self.pos[0] + delta[0]
            row = self.pos[1] + delta[1]
            if self.pointInOwnCastle((col, row)) and (
                            board[col][row] is None or board[col][row].color != self.color):
                res.append((col, row))
        return res

    def netRep(self):
        return 3 if self.color else -3

    def __repr__(self):
        letter = "K" if self.color else "k"
        place = chr(ord('a') + self.pos[0]) + chr(ord('A') + self.pos[1])
        return letter + place


class ChadGame:
    def __init__(self, boardString, playerTurn):
        self.turn = playerTurn
        self.black = []
        self.white = []
        self.kings = 0
        self.board = self.readBoard(boardString)



    def readBoard(self, bs):
        pieces = {(ord(bs[i + 1]) - ord('a'), ord(bs[i + 2]) - ord('A')): bs[i] for i in range(0, len(bs), 3)}
        return [[self.makePiece((pieces.get((i, j), None), (i, j))) for j in range(12)] for i in range(12)]

    def makePiece(self, piece):
        if piece[0] is None:
            return None
        # print("black", self.black)
        # print("white", self.white)
        color = piece[0] in {'R', 'Q', 'K'}
        if piece[0] in {'R', 'r'}:
            p = Rook(piece[1], color)
            if color:
                self.black.append(p)
            else:
                self.white.append(p)
            return p
        if piece[0] in {'Q', 'q'}:
            p = Queen(piece[1], color)
            if color:
                self.black.append(p)
            else:
                self.white.append(p)
            return p
        if piece[0] in {'K', 'k'}:
            p = King(piece[1], color)
            self.kings += 1
            if color:
                self.black.append(p)
            else:
                self.white.append(p)
            return p
        return None

    def moves(self):
        moves = []
        for p in (self.black if self.turn else self.white):
            m = p.validMoves(self.board)
            moves.extend([(p.pos, move) for move in m])
        return moves

    def getPieceAt(self,place):
        return self.board[place[0]][place[1]]

    def move(self, move):
        start = move[0]
        dest = move[1]

        movePiece = self.getPieceAt(start)
        if movePiece is None:
            return
        if self.getPieceAt(dest) is not None and self.getPieceAt(dest).type == 'k':
            self.kings -= 1

        self.board[dest[0]][dest[1]] = movePiece
        movePiece.move(dest)
        self.board[start[0]][start[1]] = None

        if movePiece.type == 'r' and movePiece.inOtherCastle():
            #print("Rook in other castle")
            self.board[dest[0]][dest[1]] = Queen(movePiece.pos, movePiece.color)

        self.turn = not self.turn

        #self.printBoard()

        return self
    def tryMove(self, move):
        game = copy.copy(self)
        game.move(move)
        assert game.networkFormat() != self.networkFormat()
        return game

    def gameOver(self):
        return (len(self.black) + len(self.white)) < 4 or self.kings != 2

    def conNetFormat(self):
        return np.array([[[p.netRep() if p is not None else 0 for p in r] for r in self.board]])

    def networkFormat(self):
        return (p.netRep() if p is not None else 0 for r in self.board for p in r)
    """"
    def __copy__(self):
        gameState = self.__repr__()
        return ChadGame(gameState[1:], True if gameState[1] == 'B' else False)
    """
    def __repr__(self):
        res = ""
        for p in self.black:
            res += p.__repr__()
        for p in self.white:
            res += p.__repr__()
        return ('B' if self.turn else 'W') + res

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):


        whiteSame = {s for s in {p.__repr__() for p in self.white} if s not in {p.__repr__() for p in other.white}}
        blackSame = {s for s in {p.__repr__() for p in self.black} if s not in {p.__repr__() for p in other.black}}

        return len(whiteSame) + len(blackSame) == 0 and self.turn == other.turn

    def __hash__(self):
        return hash((tuple(self.white), tuple(self.black), self.turn))

    def printBoard(self):
        for r in range(11,-1,-1):
            for s in range(12):
                if self.board[r][s] is None:
                    if (r,s) in whiteWall or (r,s) in blackWall:
                        print("   ", end="|")
                    else:
                        print(" - ", end="|")
                else:
                    print(" {} ".format(self.board[r][s].__repr__()[0]), end="|")
            print("")
        print()

def newGame():
    return ChadGame("rcCrcDrcErdCkdDrdEreCreDreERhHRhIRhJRiHKiIRiJRjHRjIRjJ", False)

if __name__ == '__main__':

    game = ChadGame("riB",False)
    game.printBoard()


    print(game.conNetFormat())
    print(game.conNetFormat().shape)
