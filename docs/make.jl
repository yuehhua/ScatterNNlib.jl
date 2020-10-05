using ScatterNNlib
using Documenter

makedocs(
    modules = [ScatterNNlib],
    authors = "Yueh-Hua Tu",
    repo = "https://github.com/yuehhua/ScatterNNlib.jl/blob/{commit}{path}#L{line}",
    sitename = "ScatterNNlib",
    format = Documenter.HTML(),
    pages=[
        "Home" => "index.md",
        "Manual" =>
               ["Scatter operations" => "manual/scatter.md",
               ]
    ],
)

deploydocs(repo = "github.com/yuehhua/ScatterNNlib.jl")
