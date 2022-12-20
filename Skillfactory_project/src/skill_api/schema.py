from pydantic import BaseModel, Field


class ModelTrain(BaseModel):
    Data: list[dict]


class ModelOutp(BaseModel):
    Training_time: str
    Status: str
    Message: str
    accuracy_score: str
    f1_score: str


feature_names = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


class Wine(BaseModel):
    fixed_acidity: float = Field(
        ..., ge=0, description="grams per cubic decimeter of tartaric acid"
    )
    volatile_acidity: float = Field(
        ..., ge=0, description="grams per cubic decimeter of acetic acid"
    )
    citric_acid: float = Field(..., ge=0, description="grams per cubic decimeter of citric acid")
    residual_sugar: float = Field(
        ..., ge=0, description="grams per cubic decimeter of residual sugar"
    )
    chlorides: float = Field(..., ge=0, description="grams per cubic decimeter of sodium chloride")
    free_sulfur_dioxide: float = Field(
        ..., ge=0, description="milligrams per cubic decimeter of free sulfur dioxide"
    )
    total_sulfur_dioxide: float = Field(
        ..., ge=0, description="milligrams per cubic decimeter of total sulfur dioxide"
    )
    density: float = Field(..., ge=0, description="grams per cubic meter")
    pH: float = Field(..., ge=0, lt=14, description="measure of the acidity or basicity")
    sulphates: float = Field(
        ..., ge=0, description="grams per cubic decimeter of potassium sulphate"
    )
    alcohol: float = Field(..., ge=0, le=100, description="alcohol percent by volume")


class Rating(BaseModel):
    quality: float = Field(
        ...,
        ge=0,
        le=1,
        description="wine quality grade ranging from 0 (bad wine) to 1 (good wine)",
    )


