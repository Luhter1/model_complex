from .Interface import BRModel
from .Models import (
    AgeGroupBRModel, 
    TotalBRModel,
    # StrainBRModel,
    # PairwiseModel,
    # SIRNetworkModel,
    # SEIRNetworkModel,
)


class FactoryBRModel:
    @classmethod
    def total(self) -> BRModel:
        return TotalBRModel()

    @classmethod
    def age_group(self) -> BRModel:
        return AgeGroupBRModel()

    # Arguments for models using networks passed for
    # creation of graph once during initialization.
    # @classmethod
    # def pairwise(population_size: int, m_vertices: int) -> PairwiseModel:
    #     return PairwiseModel(population_size, m_vertices)
    # @classmethod
    # def sir_network(population_size: int, m_vertices: int) -> SIRNetworkModel:
    #     return SIRNetworkModel(population_size, m_vertices)
    # @classmethod
    # def seir_network(population_size: int, m_vertices: int) -> SEIRNetworkModel:
    #     return SEIRNetworkModel(population_size, m_vertices)