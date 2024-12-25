from .Interface import Model

from .Models import (
    TotalBRModel,
    AgeGroupBRModel
    # StrainBRModel,
    # PairwiseModel,
    # SIRNetworkModel,
    # SEIRNetworkModel,
)



class FactoryBRModel:
    @classmethod
    def total() -> Model:
        return TotalBRModel()

    @classmethod
    def age_group() -> Model:
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