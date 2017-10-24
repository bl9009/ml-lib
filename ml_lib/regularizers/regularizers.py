"""Implementation of regularization methods."""

import ..numpy_utils as np_utils

class Regularizer(object):

    def __init__(self, alpha=0.):
        self.alpha = alpha

    def penalty(model):
        pass

    def penalty_derived(model):
        pass

class ElasticNet(Regularizer):
    """Elastic net regularization based on LASSO and Ridge regularization.

    Attributes:
        alpha: Regularization rate.
        l1_ratio: Ratio for l1 (LASSO) and l2 (Ridge) penalties
    """
    def init(self, alpha=0., l1_ratio=1.):
        super(ElasticNet, self).__init__(alpha)

        self.l1_ratio = l1_ratio

        self.l1_reg = Lasso(self.alpha)
        self.l2_reg = Ridge(self.alpha)

    def penalty(model):
        """Calculate penalty based on l1 and l2 penalties.

        Args:
            model: Model to calculate penalties for.

        Returns:
            Penalty based on l1 and l2 penalties.
        """
        l1 = self.l1_reg.penalty(model)
        l2 = self.l2_reg.penalty(model)

        return self.elastic_net(l1, l2)

    def penalty_derived(model):
        """Calculate penalty based on partial derivatives of l1 and l2
        penalties. This is usually used for optimization algorithms like
        gradient descent.

        Args:
            model: Model to calculate penalties for.

        Retruns:
            Penalty based on l1 and l2 derived penalties.
        """
        l1 = self.l1_reg.penalty_derived(model)
        l2 = self.l2_reg.penalty_derived(model)

        return self.elastic_net(l1, l2)

    def elastic_net(l1=0., l2=0.):
        return self.l1_ratio * l1 + (1 - self.l1_ratio) * l2

class Lasso(Regularizer):
    """LASSO regularization, regularizes model by adding l1 norm
    of vector theta.

    Attributes:
        alpha: Regularization rate.
    """
    def penalty(model):
        return self.alpha * abs(model.theta)

    def penalty_derived(model):
        return self.alpha * self.sign(model.theta)

    @np_utils.vectorize
    def sign(theta):
        """Calculates subgradient derivative for LASSO penalty."""
        if theta > 0:
            return 1
        if theta == 0:
            return 0
        if theta < 0:
            return -1

class Ridge(Regularizer):
    """Ridge regularization, regularizes model by adding l2 norm
    of vector theta.

    Attributes:
        alpha: Regularization rate.
    """
    def penalty(model):
        return self.alpha * model.theta ** 2

    def penalty_derived(model):
        return self.alpha * 2 * model.theta
