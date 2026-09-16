use std::{env, fs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 5 {
        return Err("usage: cvm-policy-eval POLICY INPUT DATA QUERY".into());
    }
    let mut engine = regorus::Engine::new();
    engine.add_policy_from_file(&args[1])?;
    engine.set_input_json(&fs::read_to_string(&args[2])?)?;
    engine.add_data(regorus::Value::from_json_str(&fs::read_to_string(&args[3])?)?)?;
    println!("{}", engine.eval_bool_query(args[4].clone(), false)?);
    Ok(())
}
