# Download the CCLE and gCSI PSets from PharmacoDB and export their contents as CSV.
#
# Downloads to  data/pharmacodb/psets/*.rds
# Writes tables to  data/pharmacodb/raw/<pset>/
#
# Run scripts/extract_pharmacodb.R next to assemble the sensitivity and
# expression tables used by src/data/build_cross_eval_matrices.py.

library(PharmacoGx)
library(data.table)

OUT_DIR  <- "data/pharmacodb/raw"
PSET_DIR <- "data/pharmacodb/psets"

dir.create(OUT_DIR,  recursive = TRUE, showWarnings = FALSE)
dir.create(PSET_DIR, recursive = TRUE, showWarnings = FALSE)

flatten_df <- function(df) {
  for (col in colnames(df)) {
    if (is.list(df[[col]])) {
      df[[col]] <- sapply(df[[col]], function(x) {
        if (is.null(x) || length(x) == 0) NA_character_
        else paste(x, collapse = "|")
      })
    }
  }
  df
}

# PSet Name -> local RDS filename
PSET_FILES <- list(
  "CCLE_2015"         = "CCLE_2015.rds",
  "CCLE_2019"         = "CCLE_2019.rds",
  "CTRPv2_2015"       = "CTRPv2_2015.rds",
  "FIMM_2016"         = "FIMM_2016.rds",
  "GDSC_2020(v1-8.2)" = "GDSC_2020_v1-8_2_.rds",
  "GDSC_2020(v2-8.2)" = "GDSC_2020_v2-8_2_.rds",
  "GRAY_2017"         = "GRAY_2017.rds",
  "NCI60_2021"        = "NCI60_2021.rds",
  "PRISM_2020"        = "PRISM_2020.rds",
  "UHNBreast_2019"    = "UHNBreast_2019.rds",
  "gCSI_2019"         = "gCSI_2019.rds"
)

target_names <- names(PSET_FILES)

cat(sprintf("Targeting %d local PSets:\n", length(target_names)))
cat(paste(target_names, collapse = ", "), "\n")

for (pset_name in target_names) {
  cat(sprintf("\n[%s] Loading local RDS...\n", pset_name))

  safe_name <- gsub("[^A-Za-z0-9_\\-]", "_", pset_name)
  pset_dir  <- file.path(OUT_DIR, safe_name)
  dir.create(pset_dir, recursive = TRUE, showWarnings = FALSE)

  writeLines(pset_name, file.path(pset_dir, "pset_name.txt"))

  rds_file <- PSET_FILES[[pset_name]]
  rds_path <- file.path(PSET_DIR, rds_file)

  tryCatch({
    if (!file.exists(rds_path)) {
      stop(sprintf("RDS file not found: %s", rds_path))
    }

    cat(sprintf("  Reading: %s\n", rds_path))
    pset <- readRDS(rds_path)

    # -- 1. Drug info -----------------------------------------------------
    cat("  Exporting drug info...\n")
    drug_df      <- flatten_df(as.data.frame(drugInfo(pset)))
    drug_df$pset <- pset_name
    fwrite(drug_df, file.path(pset_dir, "drug_info.csv"))
    cat(sprintf("    %d drugs\n", nrow(drug_df)))

    # -- 2. Cell line info ------------------------------------------------
    cat("  Exporting cell info...\n")
    cell_df      <- flatten_df(as.data.frame(cellInfo(pset)))
    cell_df$pset <- pset_name
    fwrite(cell_df, file.path(pset_dir, "cell_info.csv"))
    cat(sprintf("    %d cell lines\n", nrow(cell_df)))

    # -- 3. Drug sensitivity ----------------------------------------------
    measures <- sensitivityMeasures(pset)
    cat(sprintf("  Available sensitivity measures: %s\n",
                paste(measures, collapse = ", ")))

    sensitivity_candidates <- c(
      "aac_recomputed",
      "AAC",
      "auc_recomputed",
      "ic50_recomputed",
      "IC50",
      "AUC"
    )

    for (measure in sensitivity_candidates) {
      if (measure %in% measures) {
        cat(sprintf("  Summarizing %s...\n", measure))

        summ <- tryCatch(
          summarizeSensitivityProfiles(
            pset,
            sensitivity.measure = measure,
            summary.stat        = "median",
            verbose             = FALSE
          ),
          error = function(e) {
            cat(sprintf("    WARN sensitivity %s: %s\n", measure, e$message))
            NULL
          }
        )

        if (!is.null(summ)) {
          summ_df           <- flatten_df(as.data.frame(summ))
          summ_df$drug_name <- rownames(summ_df)

          out_f <- file.path(pset_dir, sprintf("sensitivity_%s.csv", measure))
          fwrite(summ_df, out_f)

          cat(sprintf("    Saved: %d drugs x %d cells -> %s\n",
                      nrow(summ_df), ncol(summ_df) - 1, basename(out_f)))
        }
      }
    }

    # -- 4. Molecular profiles --------------------------------------------
    cat("  Checking molecular profiles...\n")

    mol_types <- tryCatch(
      names(molecularProfilesSlot(pset)),
      error = function(e) {
        cat(sprintf("    WARN molecularProfilesSlot: %s\n", e$message))
        character(0)
      }
    )

    cat(sprintf("  Molecular types: %s\n", paste(mol_types, collapse = ", ")))

    # -- 5. Gene Expression -----------------------------------------------
    expr_candidates <- c(
      "rna",
      "rnaseq",
      "rna.seq",
      "RNA",
      "RNASeq",
      "Kallisto_0.46.1.rnaseq",
      "rnaseq_counts",
      "microarray_normalized",
      "rna_microarray",
      "rna_seq"
    )

    expr_saved <- FALSE

    for (mol_type in expr_candidates) {
      if (mol_type %in% mol_types) {
        cat(sprintf("  Exporting expression (%s)...\n", mol_type))

        expr <- tryCatch(
          summarizeMolecularProfiles(
            pset,
            mDataType    = mol_type,
            summary.stat = "median",
            verbose      = FALSE
          ),
          error = function(e) {
            cat(sprintf("    WARN expression %s: %s\n", mol_type, e$message))
            NULL
          }
        )

        if (!is.null(expr)) {
          expr_mat <- tryCatch(
            assay(expr),
            error = function(e) {
              cat(sprintf("    WARN assay expression %s: %s\n", mol_type, e$message))
              NULL
            }
          )

          if (!is.null(expr_mat)) {
            cat(sprintf("    Matrix: %d genes x %d cells\n",
                        nrow(expr_mat), ncol(expr_mat)))

            expr_df           <- flatten_df(as.data.frame(t(expr_mat)))
            expr_df$cell_name <- rownames(expr_df)

            fwrite(expr_df, file.path(pset_dir, "expression.csv"))
            writeLines(mol_type, file.path(pset_dir, "expression_mol_type.txt"))

            cat("    Saved expression.csv\n")
            expr_saved <- TRUE
            break
          }
        }
      }
    }

    if (!expr_saved) {
      cat("  No expression profile exported.\n")
    }

    # -- 6. Mutation ------------------------------------------------------
    mutation_candidates <- c(
      "mutation",
      "mutation.snp",
      "mut",
      "snv",
      "mutation_call",
      "mutation_calls"
    )

    mut_saved <- FALSE

    for (mol_type in mutation_candidates) {
      if (mol_type %in% mol_types) {
        cat(sprintf("  Exporting mutation (%s)...\n", mol_type))

        mut <- tryCatch(
          summarizeMolecularProfiles(
            pset,
            mDataType    = mol_type,
            summary.stat = "or",
            verbose      = FALSE
          ),
          error = function(e) {
            cat(sprintf("    WARN mutation %s: %s\n", mol_type, e$message))
            NULL
          }
        )

        if (!is.null(mut)) {
          mut_mat <- tryCatch(
            assay(mut),
            error = function(e) {
              cat(sprintf("    WARN assay mutation %s: %s\n", mol_type, e$message))
              NULL
            }
          )

          if (!is.null(mut_mat)) {
            mut_df           <- flatten_df(as.data.frame(t(mut_mat)))
            mut_df$cell_name <- rownames(mut_df)

            fwrite(mut_df, file.path(pset_dir, "mutation.csv"))
            writeLines(mol_type, file.path(pset_dir, "mutation_mol_type.txt"))

            cat(sprintf("    Saved mutation.csv: %d cells x %d genes\n",
                        nrow(mut_df), ncol(mut_df) - 1))
            mut_saved <- TRUE
            break
          }
        }
      }
    }

    if (!mut_saved) {
      cat("  No mutation profile exported.\n")
    }

    cat(sprintf("  [DONE] %s -> %s\n", pset_name, safe_name))

  }, error = function(e) {
    cat(sprintf("  [ERROR] %s: %s\n", pset_name, e$message))
  })
}

cat("\n=== Export complete ===\n")
cat(sprintf("Output: %s\n", OUT_DIR))
cat("Next: Rscript scripts/extract_pharmacodb.R\n")