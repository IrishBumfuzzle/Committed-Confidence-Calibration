# Build command: `make` (needs tectonic; fetches TeX packages on first run).
main.pdf: main.tex refs.bib ../artifacts/probe/figures/c2_crossover_ft110m-v3-visible.pdf ../artifacts/t12/figures/t12_matched_speed.pdf
	tectonic main.tex

.PHONY: clean
clean:
	rm -f main.pdf main.aux main.bbl main.blg main.log main.out
