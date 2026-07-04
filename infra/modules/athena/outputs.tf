output "workgroup_names" {
  description = "Athena workgroup names."
  value = {
    for key, wg in aws_athena_workgroup.this : key => wg.name
  }
}

output "agent_readonly_workgroup_name" {
  description = "Agent read-only Athena workgroup name."
  value       = aws_athena_workgroup.this["agent_readonly"].name
}