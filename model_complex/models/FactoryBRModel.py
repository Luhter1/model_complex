from .Interface import Model
from .Models import AgeGroupBRModel, TotalBRModel


class FactoryBRModel:
    @classmethod
    def total(self) -> Model:
        return TotalBRModel()

    @classmethod
    def age_group(self) -> Model:
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
