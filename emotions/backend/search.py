"""
Search functionality on MM results.
"""
from typing import List
from typeguard import check_type
from typing_inspect import get_origin
from typing_extensions import Literal
import operator as op
from emotions.constants import THERAPIST, PATIENT, SPEAKER, UTTERANCE


# Type Definitions.
TARGET_ARGS = [SPEAKER]
MM_ARGS = ["mm_a", "mm_b", "mm_c"]
STR_ARGS = [THERAPIST, PATIENT]

# Exp_Operator = str
QUERY_OPERATORS = [">", "<", ">=", "<=", "=", "=="]
EXP_OPERATORS = ["or", "and", "not"]


def expression_operators():
    """
    Environment.
    """

    def validate_args(args):
        target_arg = None
        mm_arg = None
        filter_arg = None
        for arg in args:
            if arg in TARGET_ARGS:
                if target_arg is not None:
                    raise ValueError(
                        "Redundant values for target arg: %s, %s."
                        % (target_arg, arg)
                    )
                target_arg = arg

            elif arg in MM_ARGS:
                if mm_arg is not None:
                    raise ValueError(
                        "Redundant values for mm arg: %s, %s." % (mm_arg, arg)
                    )
                mm_arg = arg

            elif isinstance(arg, float) or isinstance(arg, str):
                if filter_arg is not None:
                    raise ValueError(
                        "Redundant values for filter arg: %s, %s."
                        % (str(filter_arg), str(arg))
                    )
                filter_arg = arg

            else:
                raise ValueError("Invalid type for arg: %s." % arg)

        if filter_arg is None:
            raise ValueError("Need to specify filter arg.")
        if target_arg is None and mm_arg is None:
            raise ValueError("Must specify either mm_arg or target_arg.")
        if target_arg is not None and mm_arg is not None:
            raise ValueError("Can't specify both target_arg and mm_arg.")

        if target_arg is not None and not isinstance(filter_arg, str):
            raise ValueError(
                "Invalid argument for filter_arg: %s" % str(filter_arg)
            )
        if mm_arg is not None and not isinstance(filter_arg, float):
            raise ValueError(
                "Invalid argument for filter_arg: %s" % str(filter_arg)
            )
        return target_arg or mm_arg, filter_arg

    def validate_numeric_args(target_arg, filter_arg):
        """
        Validate args for numeric operators.
        """
        if target_arg in TARGET_ARGS:
            raise ValueError(
                "Invalid target_arg for 'greater_than()': %s." % target_arg
            )

        if not isinstance(filter_arg, float):
            raise ValueError(
                "Invalid filter_arg for 'greater_than()': %s."
                % str(filter_arg)
            )

    def get_values(data, target_arg):
        if target_arg in TARGET_ARGS:
            return [x["speaker"] for x in data]
        if target_arg in MM_ARGS:
            return [x["micromodels"][target_arg] for x in data]
        raise ValueError(
            "How did we get here? Invalid instance for target_arg: %s."
            % type(target_arg)
        )

    def greater_than(data, args):
        target_arg, filter_arg = validate_args(args)
        validate_numeric_args(target_arg, filter_arg)

        _data = get_values(data, target_arg)
        return [idx for idx, x in enumerate(_data) if x > filter_arg]

    def less_than(data, args):
        target_arg, filter_arg = validate_args(args)
        validate_numeric_args(target_arg, filter_arg)

        _data = get_values(data, target_arg)
        return [idx for idx, x in enumerate(_data) if x < filter_arg]

    def greater_than_or_equal(data, args):
        target_arg, filter_arg = validate_args(args)
        validate_numeric_args(target_arg, filter_arg)

        _data = get_values(data, target_arg)
        return [idx for idx, x in enumerate(_data) if x >= filter_arg]

    def less_than_or_equal(data, args):
        target_arg, filter_arg = validate_args(args)
        validate_numeric_args(target_arg, filter_arg)

        _data = get_values(data, target_arg)
        return [idx for idx, x in enumerate(_data) if x <= filter_arg]

    def equal(data, args):
        target_arg, filter_arg = validate_args(args)
        if target_arg in TARGET_ARGS:
            if filter_arg not in STR_ARGS:
                raise ValueError(
                    "Invalid filter_arg for 'equal' operator: %s" % filter_arg
                )
        if target_arg in MM_ARGS:
            if not isinstance(filter_arg, float):
                raise ValueError(
                    "Invalid filter_arg for 'equal' operator: %s" % filter_arg
                )

        _data = get_values(data, target_arg)
        return [idx for idx, x in enumerate(_data) if x == filter_arg]

    def or_op(data, _args):

        ret = []
        for _arg in _args:
            if not isinstance(_arg, list):
                breakpoint()
                raise ValueError("Invalid arg type received.")
            ret.extend(_arg)
        return list(set(ret))

    def and_op(data, _args):
        ret = []
        if len(_args) < 1:
            raise ValueError("Empty list of args.")
        ret = _args[0]
        if not isinstance(ret, list):
            raise ValueError("First argument is not a valid type, should be a list.")

        for _arg in _args[1:]:
            if not isinstance(_arg, list):
                raise ValueError("Argument is not a valid type, should be a list.")
            ret = [x for x in ret if x in _arg]
        return ret

    env = {
        ">": greater_than,
        "<": less_than,
        ">=": greater_than_or_equal,
        "<=": less_than_or_equal,
        "=": equal,
        "==": equal,
        "or": or_op,
        "and": and_op,
    }
    return env


ops = expression_operators()


def resolve(query):
    """
    :query:
    """
    return query


def atom(token: str):
    try:
        return float(token)
    except ValueError:
        return token


def read_from_tokens(tokens):
    if len(tokens) == 0:
        raise SyntaxError("Unexpected EOF.")

    token = tokens.pop(0)
    if token == "(":
        L = []
        while tokens[0] != ")":
            L.append(read_from_tokens(tokens))
        tokens.pop(0)
        return L
    elif token == ")":
        raise SyntaxError("Unexpected ')'.")

    else:
        return atom(token)


def parse_query(query):
    """
    Parse query.
    :query: str
    return: List of Query objects.
    """
    paren_idxs = [idx for idx, char in enumerate(query) if char in ["(", ")"]]
    if len(paren_idxs) == 0:
        query = "( " + query + " )"

    spaced_query = ""
    for idx, char in enumerate(query):
        if idx in paren_idxs:
            spaced_query += " "
        spaced_query += char
        if idx in paren_idxs:
            spaced_query += " "

    tokens = spaced_query.split()
    return read_from_tokens(tokens)


def evaluate(exp, env, data):
    """
    Evaluate an expression.
    """
    if isinstance(exp, str) or isinstance(exp, float):
        if exp in QUERY_OPERATORS + EXP_OPERATORS:
            return env[exp]
        return exp

    operator = evaluate(exp[0], env, data)
    _args = [evaluate(arg, env, data) for arg in exp[1:]]
    return operator(data, _args)


if __name__ == "__main__":

    test_data = [
        {
            "utterance": "testing 1",
            "speaker": THERAPIST,
            "micromodels": {
                "mm_a": 0.7,
                "mm_b": 0.6,
                "mm_c": 0.5,
                "mm_d": 0.4,
                "mm_e": 0.3,
                "mm_f": 0.2,
            },
        },
        {
            "utterance": "testing 2",
            "speaker": PATIENT,
            "micromodels": {
                "mm_a": 0.9,
                "mm_b": 0.8,
                "mm_c": 0.7,
                "mm_d": 0.1,
                "mm_e": 0.2,
                "mm_f": 0.3,
            },
        },
        {
            "utterance": "testing 3",
            "speaker": THERAPIST,
            "micromodels": {
                "mm_a": 0.1,
                "mm_b": 0.2,
                "mm_c": 0.1,
                "mm_d": 0.2,
                "mm_e": 0.1,
                "mm_f": 0.2,
            },
        },
        {
            "utterance": "testing 4",
            "speaker": PATIENT,
            "micromodels": {
                "mm_a": 0.5,
                "mm_b": 0.7,
                "mm_c": 0.2,
                "mm_d": 0.5,
                "mm_e": 0.7,
                "mm_f": 0.2,
            },
        },
        {
            "utterance": "testing 5",
            "speaker": THERAPIST,
            "micromodels": {
                "mm_a": 0.7,
                "mm_b": 0.8,
                "mm_c": 0.9,
                "mm_d": 0.7,
                "mm_e": 0.8,
                "mm_f": 0.9,
            },
        },
    ]
    # parsed_query = parse_query("((== speaker therapist) or (utterance == this))")
    parsed_query = parse_query("> mm_a 0.5")
    x = evaluate(parsed_query, ops, test_data)
    assert x == [0, 1, 4]

    parsed_query = parse_query("== speaker " + THERAPIST)
    y = evaluate(parsed_query, ops, test_data)
    assert y == [0, 2, 4]

    parsed_query = parse_query("(or (== speaker %s) (> mm_a 0.5))" % THERAPIST)
    assert parsed_query == ["or", ["==", "speaker", THERAPIST], [">", "mm_a", 0.5]]

    z = evaluate(parsed_query, ops, test_data)
    assert z == [0, 1, 2, 4]

    parsed_query = parse_query("(and (== speaker %s) (> mm_a 0.5))" % THERAPIST)
    zz = evaluate(parsed_query, ops, test_data)
    assert zz == [0, 4]

    parsed_query = parse_query("(or (and (== speaker %s) (> mm_a 0.5)) (<= mm_b 0.5))" % THERAPIST)
    zz = evaluate(parsed_query, ops, test_data)
    assert zz == [0, 2, 4]
    print("Hmm...")
