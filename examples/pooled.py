def makePooledSpace(base, *args, **kwargs):
    """ Pooled space provides a optimization in reusing vectors and returning them from empty.
    """
    BaseType = type(base)
    BaseVectorType = BaseType.Vector

    class PooledSpace(BaseType):
        def __init__(self, base, *args, **kwargs):
            self._pool = []
            self._poolMaxSize = kwargs.pop('maxPoolSize', 1)
            self._base = base

        def empty(self):
            if self._pool:
                return self._pool.pop()
            else:
                return BaseType.empty(self)

        def __getattr__(self, name):
            return getattr(self._base, name)

        def __str__(self):
            return "PooledSpace(" + str(self._base) + ", Pool size:" + str(len(self._pool)) + ")"

        class Vector(BaseVectorType):
            def __del__(self):
                if len(self.space._pool) < self.space._poolMaxSize:
                    self.space._pool.append(self)
                else:
                    pass#TODO BaseVectorType.__del__(self)

    return PooledSpace(base, *args, **kwargs)