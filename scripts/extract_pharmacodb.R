# Export sensitivity and expression tables from the downloaded PSets.
#
# Reads  data/pharmacodb/raw/<pset>/  and writes the per-dataset tables that
# src/data/build_cross_eval_matrices.py turns into evaluation matrices.

suppressPackageStartupMessages({
  library(PharmacoGx)
  library(data.table)
  library(SummarizedExperiment)
})

PSET_DIR <- "data/pharmacodb/psets"
OUT_DIR  <- "data/pharmacodb/raw"

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

flatten_df <- function(df) {
  df <- as.data.frame(df, stringsAsFactors = FALSE)

  for (col in colnames(df)) {
    if (is.list(df[[col]])) {
      df[[col]] <- sapply(df[[col]], function(x) {
        if (is.null(x) || length(x) == 0) {
          NA_character_
        } else {
          paste(as.character(x), collapse = "|")
        }
      })
    }
  }

  df
}

safe_write_csv <- function(obj, out_file, label) {
  tryCatch({
    df <- flatten_df(as.data.frame(obj))
    fwrite(df, out_file)
    cat(sprintf("  %s: %d rows x %d cols\n", label, nrow(df), ncol(df)))
    TRUE
  }, error = function(e) {
    cat(sprintf("  [WARN] failed to write %s: %s\n", label, e$message))
    FALSE
  })
}

get_slot_safe <- function(obj, slot_name) {
  if (slot_name %in% slotNames(obj)) {
    return(slot(obj, slot_name))
  }
  NULL
}

write_slot_object <- function(obj, out_file, label) {
  if (is.null(obj)) return(FALSE)

  tryCatch({
    df <- flatten_df(as.data.frame(obj))
    fwrite(df, out_file)
    cat(sprintf("  %s: %d rows x %d cols\n", label, nrow(df), ncol(df)))
    TRUE
  }, error = function(e) {
    cat(sprintf("  [WARN] %s direct slot write failed: %s\n", label, e$message))
    FALSE
  })
}

write_assay_matrix <- function(mat, out_file, row_id_name, label, transpose = FALSE) {
  tryCatch({
    if (transpose) {
      df <- as.data.frame(t(as.matrix(mat)), stringsAsFactors = FALSE)
    } else {
      df <- as.data.frame(as.matrix(mat), stringsAsFactors = FALSE)
    }

    df[[row_id_name]] <- rownames(df)
    df <- df[, c(row_id_name, setdiff(colnames(df), row_id_name)), drop = FALSE]

    fwrite(df, out_file)
    cat(sprintf("  %s: %d rows x %d cols\n", label, nrow(df), ncol(df)))
    TRUE
  }, error = function(e) {
    cat(sprintf("  [WARN] %s matrix write failed: %s\n", label, e$message))
    FALSE
  })
}

extract_drug_info <- function(pset, pset_dir, pset_name) {
  out_file <- file.path(pset_dir, "drug_info.csv")

  ok <- tryCatch({
    drug_df <- flatten_df(as.data.frame(drugInfo(pset)))
    drug_df$pset <- pset_name
    fwrite(drug_df, out_file)
    cat(sprintf("  drug_info.csv: %d drugs x %d cols\n", nrow(drug_df), ncol(drug_df)))
    TRUE
  }, error = function(e) {
    cat(sprintf("  [WARN] drugInfo() failed: %s\n", e$message))
    FALSE
  })

  if (ok) return(TRUE)

  drug_slot <- get_slot_safe(pset, "drug")
  ok <- write_slot_object(drug_slot, out_file, "drug_info.csv from @drug")

  if (!ok && !is.null(drug_slot)) {
    cat("  [INFO] drug slot class: ", paste(class(drug_slot), collapse = ", "), "\n")
    cat("  [INFO] drug slot names: ", paste(names(drug_slot), collapse = ", "), "\n")
  }

  ok
}

extract_cell_info <- function(pset, pset_dir, pset_name) {
  out_file <- file.path(pset_dir, "cell_info.csv")

  ok <- tryCatch({
    cell_df <- flatten_df(as.data.frame(cellInfo(pset)))
    cell_df$pset <- pset_name
    fwrite(cell_df, out_file)
    cat(sprintf("  cell_info.csv: %d cells x %d cols\n", nrow(cell_df), ncol(cell_df)))
    TRUE
  }, error = function(e) {
    cat(sprintf("  [WARN] cellInfo() failed: %s\n", e$message))
    FALSE
  })

  if (ok) return(TRUE)

  cell_slot <- get_slot_safe(pset, "cell")
  ok <- write_slot_object(cell_slot, out_file, "cell_info.csv from @cell")

  if (!ok && !is.null(cell_slot)) {
    cat("  [INFO] cell slot class: ", paste(class(cell_slot), collapse = ", "), "\n")
    cat("  [INFO] cell slot names: ", paste(names(cell_slot), collapse = ", "), "\n")
  }

  ok
}

extract_sensitivity <- function(pset, pset_dir) {
  measures <- tryCatch(sensitivityMeasures(pset), error = function(e) character(0))
  cat(sprintf("  Sensitivity measures: %s\n", paste(measures, collapse = ", ")))

  wrote_any <- FALSE

  for (measure in measures) {
    ok <- tryCatch({
      summ <- summarizeSensitivityProfiles(
        pset,
        sensitivity.measure = measure,
        summary.stat = "median",
        verbose = FALSE
      )

      summ_df <- flatten_df(as.data.frame(summ))
      summ_df$drug_name <- rownames(summ_df)

      out_f <- file.path(pset_dir, sprintf("sensitivity_%s.csv", make.names(measure)))
      fwrite(summ_df, out_f)

      cat(sprintf("  sensitivity_%s.csv: %d rows x %d cols\n",
                  measure, nrow(summ_df), ncol(summ_df)))
      TRUE
    }, error = function(e) {
      cat(sprintf("  [WARN] summarizeSensitivityProfiles(%s) failed: %s\n", measure, e$message))
      FALSE
    })

    if (ok) wrote_any <- TRUE
  }

  if (wrote_any) return(TRUE)

  sens_slot <- get_slot_safe(pset, "sensitivity")
  if (is.null(sens_slot)) {
    cat("  [WARN] no @sensitivity slot\n")
    return(FALSE)
  }

  cat("  [INFO] Trying direct @sensitivity extraction...\n")
  cat("  [INFO] sensitivity slot class: ", paste(class(sens_slot), collapse = ", "), "\n")
  cat("  [INFO] sensitivity slot names: ", paste(names(sens_slot), collapse = ", "), "\n")

  if ("profiles" %in% names(sens_slot)) {
    prof <- sens_slot$profiles

    if (is.list(prof)) {
      for (nm in names(prof)) {
        out_f <- file.path(pset_dir, sprintf("sensitivity_%s.csv", make.names(nm)))
        ok <- write_assay_matrix(prof[[nm]], out_f, "drug_name", paste0("sensitivity_", nm), transpose = FALSE)
        if (ok) wrote_any <- TRUE
      }
    } else {
      out_f <- file.path(pset_dir, "sensitivity_profiles.csv")
      ok <- write_assay_matrix(prof, out_f, "drug_name", "sensitivity_profiles", transpose = FALSE)
      if (ok) wrote_any <- TRUE
    }
  }

  if (!wrote_any) {
    ok <- write_slot_object(sens_slot, file.path(pset_dir, "sensitivity_raw.csv"), "sensitivity_raw.csv")
    if (ok) wrote_any <- TRUE
  }

  wrote_any
}

extract_molecular_profiles <- function(pset, pset_dir) {
  mol_slot <- tryCatch(molecularProfilesSlot(pset), error = function(e) NULL)

  if (is.null(mol_slot)) {
    mol_slot <- get_slot_safe(pset, "molecularProfiles")
  }

  if (is.null(mol_slot)) {
    cat("  [WARN] no molecularProfiles slot\n")
    return(FALSE)
  }

  mol_types <- names(mol_slot)
  cat(sprintf("  Molecular types: %s\n", paste(mol_types, collapse = ", ")))

  if (length(mol_types) == 0) return(FALSE)

  expression_candidates <- c(
    "rna",
    "rnaseq",
    "rna.seq",
    "RNA",
    "RNASeq",
    "Kallisto_0.46.1.rnaseq",
    "Kallisto_0.46.1.rnaseq.counts",
    "rnaseq_counts",
    "microarray_normalized"
  )

  mutation_candidates <- c(
    "mutation",
    "mutation.snp",
    "mut",
    "snv"
  )

  wrote_expr <- FALSE
  wrote_mut  <- FALSE

  # Expression
  for (mol_type in expression_candidates) {
    if (!(mol_type %in% mol_types)) next

    out_f <- file.path(pset_dir, "expression.csv")

    ok <- tryCatch({
      expr <- summarizeMolecularProfiles(
        pset,
        mDataType = mol_type,
        summary.stat = "median",
        verbose = FALSE
      )

      expr_mat <- assay(expr)
      write_assay_matrix(expr_mat, out_f, "cell_name", "expression.csv", transpose = TRUE)
    }, error = function(e) {
      cat(sprintf("  [WARN] summarizeMolecularProfiles(%s) failed: %s\n", mol_type, e$message))
      FALSE
    })

    if (!ok) {
      obj <- mol_slot[[mol_type]]

      ok <- tryCatch({
        if (inherits(obj, "SummarizedExperiment")) {
          mat <- assay(obj)
          write_assay_matrix(mat, out_f, "cell_name", "expression.csv", transpose = TRUE)
        } else if (is.matrix(obj) || is.data.frame(obj)) {
          write_assay_matrix(obj, out_f, "cell_name", "expression.csv", transpose = TRUE)
        } else {
          cat(sprintf("  [WARN] expression object class unsupported: %s\n",
                      paste(class(obj), collapse = ", ")))
          FALSE
        }
      }, error = function(e) {
        cat(sprintf("  [WARN] direct expression extraction failed: %s\n", e$message))
        FALSE
      })
    }

    if (ok) {
      wrote_expr <- TRUE
      break
    }
  }

  # Mutation
  for (mol_type in mutation_candidates) {
    if (!(mol_type %in% mol_types)) next

    out_f <- file.path(pset_dir, "mutation.csv")

    ok <- tryCatch({
      mut <- summarizeMolecularProfiles(
        pset,
        mDataType = mol_type,
        summary.stat = "or",
        verbose = FALSE
      )

      mut_mat <- assay(mut)
      write_assay_matrix(mut_mat, out_f, "cell_name", "mutation.csv", transpose = TRUE)
    }, error = function(e) {
      cat(sprintf("  [WARN] summarizeMolecularProfiles(%s) failed: %s\n", mol_type, e$message))
      FALSE
    })

    if (!ok) {
      obj <- mol_slot[[mol_type]]

      ok <- tryCatch({
        if (inherits(obj, "SummarizedExperiment")) {
          mat <- assay(obj)
          write_assay_matrix(mat, out_f, "cell_name", "mutation.csv", transpose = TRUE)
        } else if (is.matrix(obj) || is.data.frame(obj)) {
          write_assay_matrix(obj, out_f, "cell_name", "mutation.csv", transpose = TRUE)
        } else {
          cat(sprintf("  [WARN] mutation object class unsupported: %s\n",
                      paste(class(obj), collapse = ", ")))
          FALSE
        }
      }, error = function(e) {
        cat(sprintf("  [WARN] direct mutation extraction failed: %s\n", e$message))
        FALSE
      })
    }

    if (ok) {
      wrote_mut <- TRUE
      break
    }
  }

  wrote_expr || wrote_mut
}

rds_files <- list.files(PSET_DIR, pattern = "\\.rds$", full.names = TRUE)

cat(sprintf("[INFO] Found %d RDS files:\n", length(rds_files)))
for (f in rds_files) {
  cat(sprintf("  %s\n", basename(f)))
}

for (rds_path in rds_files) {
  safe_name <- tools::file_path_sans_ext(basename(rds_path))
  pset_dir  <- file.path(OUT_DIR, safe_name)
  dir.create(pset_dir, recursive = TRUE, showWarnings = FALSE)

  cat(sprintf("\n[%s] Loading RDS...\n", safe_name))

  pset <- tryCatch(readRDS(rds_path), error = function(e) {
    cat(sprintf("  [ERROR] Failed to load: %s\n", e$message))
    NULL
  })

  if (is.null(pset)) next

  pset <- tryCatch({
    updateObject(pset)
  }, error = function(e) {
    cat(sprintf("  [WARN] updateObject failed: %s\n", e$message))
    pset
  })

  cat(sprintf("  Class: %s\n", paste(class(pset), collapse = ", ")))
  cat(sprintf("  Slots: %s\n", paste(slotNames(pset), collapse = ", ")))

  pset_name <- tryCatch(name(pset), error = function(e) safe_name)
  writeLines(pset_name, file.path(pset_dir, "pset_name.txt"))
  cat(sprintf("  PSet name: %s\n", pset_name))

  extract_drug_info(pset, pset_dir, pset_name)
  extract_cell_info(pset, pset_dir, pset_name)
  extract_sensitivity(pset, pset_dir)
  extract_molecular_profiles(pset, pset_dir)

  cat(sprintf("  [DONE] %s\n", safe_name))
}

cat("\n=== Extraction complete ===\n")
cat(sprintf("Output: %s\n", OUT_DIR))
cat("Next: python src/data/build_cross_eval_matrices.py\n")