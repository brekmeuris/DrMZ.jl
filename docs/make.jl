using DrMZ
using Documenter
# using DocumenterMarkdown

# push!(LOAD_PATH,"../src/")

DocMeta.setdocmeta!(DrMZ, :DocTestSetup, :(using DrMZ); recursive=true)

makedocs(;
    modules=[DrMZ],
    authors="Brek Meuris",
    repo="https://github.com/brekmeuris/DrMZ.jl/blob/{commit}{path}#{line}",
    sitename="DrMZ.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://brekmeuris.github.io/DrMZ.jl/",
        assets=String[],
    ),
    # format = Markdown(),
    pages=[
        "Home" => "index.md",
    ],
)

# FOR HTML
deploydocs(;
    repo="github.com/brekmeuris/DrMZ.jl",
    devbranch="main",
    target="build",
#     # devurl="docs",
    branch="gh-pages",
)
