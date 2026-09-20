TEX            = pdflatex -interaction=nonstopmode
MAIN           = main
NOTES          = notes
SPEECH         = speech_notes
MAIN_PDF       = $(MAIN).pdf
NOTES_PDF      = $(NOTES).pdf
SPEECH_PDF     = $(SPEECH).pdf

# -----------------------------------------------------------------------
# Default: build both the presentation and the presenter-notes PDF
# -----------------------------------------------------------------------
all: $(MAIN_PDF) $(NOTES_PDF) $(SPEECH_PDF)

# Presentation (two passes for cross-references / ToC)
$(MAIN_PDF): $(MAIN).tex $(wildcard figures/**/*.tex figures/**/*.png)
	$(TEX) $(MAIN).tex
	$(TEX) $(MAIN).tex

# Notes-only PDF (two passes)
$(NOTES_PDF): $(NOTES).tex $(MAIN).tex $(wildcard figures/**/*.tex figures/**/*.png)
	$(TEX) $(NOTES).tex
	$(TEX) $(NOTES).tex

# Speech notes (standalone article)
$(SPEECH_PDF): $(SPEECH).tex
	$(TEX) $(SPEECH).tex
	$(TEX) $(SPEECH).tex

# Convenience aliases
presentation: $(MAIN_PDF)
notes:        $(NOTES_PDF)
speech:       $(SPEECH_PDF)

# -----------------------------------------------------------------------
# Cleanup
# NOTE: notes_thoughts_questions.pdf is a protected file and must NEVER
# be deleted by clean, fclean, or any build/cleanup process.
# -----------------------------------------------------------------------
clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb

fclean: clean
	rm -f $(MAIN_PDF) notes.pdf $(SPEECH_PDF)

re: fclean all

.PHONY: all presentation notes speech clean fclean re
