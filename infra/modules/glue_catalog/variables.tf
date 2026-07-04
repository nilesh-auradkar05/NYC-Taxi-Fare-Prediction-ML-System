variable "database_names" {
  description = "Glue database names for bronze, silver, and gold."
  type = object({
    bronze = string
    silver = string
    gold   = string
  })
}