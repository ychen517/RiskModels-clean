

class ModelFactor:
    """A factor in a risk model.
    A factor contains its name, description, factor ID and
    from/thru dates for which the factor is valid.
    """
    def __init__(self, name, description=None):
        self.name = name
        if description == None:
            self.description = name
        else:
            self.description = description
    def __eq__(self, other):
        return self.name == other.name
    def __ne__(self, other):
        return self.name != other.name
    def __lt__(self, other):
        return self.name < other.name
    def __le__(self, other):
        return self.name <= other.name
    def __gt__(self, other):
        return self.name > other.name
    def __ge__(self, other):
        return self.name >= other.name
    def __hash__(self):
        return self.name.__hash__()
    def __repr__(self):
        return 'ModelFactor(%s, %s)' % (self.name, self.description)
    
    def isLive(self, date):
        """Checks if the factor is alive as of the given date
        """
        if (not hasattr(self, 'from_dt')) or (not hasattr(self, 'thru_dt')):
            raise LookupError('Factor %s missing from and/or thru dt' % self.name)
        if self.from_dt <= date and self.thru_dt > date:
            return True
        else:
            return False

class CompositeFactor(ModelFactor):
    """A ModelFactor comprised of multiple descriptor terms.
    """
    def __init__(self, name, description):
        ModelFactor.__init__(self, name, description)
        self.descriptors = list()
    def __repr__(self):
        return 'CompositeFactor(%s, %s)' % (self.name, self.description)

class FactorDescriptor:
    """A descriptor that is part of a CompositeFactor definition.
    """
    def __init__(self, name, description):
        self.name = name
        self.description = description
    def __eq__(self, other):
        return self.name == other.name
    def __ne__(self, other):
        return self.name != other.name
    def __lt__(self, other):
        return self.name < other.name
    def __le__(self, other):
        return self.name <= other.name
    def __gt__(self, other):
        return self.name > other.name
    def __ge__(self, other):
        return self.name >= other.name
    def __hash__(self):
        return self.name.__hash__()
    def __repr__(self):
        return 'FactorDescriptor(%s, %s)' % (self.name, self.description)

