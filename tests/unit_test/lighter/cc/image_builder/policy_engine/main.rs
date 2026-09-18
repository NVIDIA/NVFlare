use std::{env, fs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 5 {
        return Err("usage: cvm-policy-eval POLICY INPUT DATA QUERY".into());
    }
    let mut engine = regorus::Engine::new();
    engine.add_policy_from_file(&args[1])?;
    engine.set_input_json(&fs::read_to_string(&args[2])?)?;
    let data = regorus::Value::from_json_str(&fs::read_to_string(&args[3])?)?;
    let references = data["reference"].clone();
    // Match Trustee v0.22's RVPS extension; fixture values stand in for RVPS.
    engine.add_extension(
        "query_reference_value".to_string(),
        1,
        Box::new(move |params: Vec<regorus::Value>| {
            let key = params[0].as_string()?;
            let value = references[&**key].clone();
            Ok(if value == regorus::Value::Undefined {
                regorus::Value::Null
            } else {
                value
            })
        }),
    )?;
    engine.add_data(data)?;
    println!("{}", engine.eval_bool_query(args[4].clone(), false)?);
    Ok(())
}
