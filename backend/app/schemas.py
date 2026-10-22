from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProductBase(BaseModel):
    name: str = Field(min_length=1, max_length=150)
    category: str = Field(min_length=1, max_length=100)
    price: float = Field(gt=0, allow_inf_nan=False)
    cost: float = Field(ge=0, allow_inf_nan=False)
    description: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def validate_cost(self):
        if self.cost > self.price:
            raise ValueError("Product cost cannot exceed its price.")
        return self


class ProductCreate(ProductBase):
    pass


class ProductUpdate(ProductBase):
    pass


class ProductResponse(ProductBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime