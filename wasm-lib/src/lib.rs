use js_sys::{Array,BigInt,Number};
use wasm_bindgen::JsValue;
use serde::Serialize;

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn hello() -> Number {
    Number::from(42)
}

#[wasm_bindgen(getter_with_clone)]
#[derive(Serialize,Clone)]
pub struct DronePos {
    pub id: u32,
    pub x: f32,
    pub y: f32,
}

#[wasm_bindgen(getter_with_clone)]
#[derive(Serialize)]
pub struct MyDummy {
    //pub drones: Vec<DronePos>,
    //pub drone: DronePos,
    pub val: u32,
}


impl MyDummy {
    pub fn new() -> MyDummy {

        let drone_poses = vec![
            DronePos{id:1, x:200.0, y:100.0},
            DronePos{id:2, x:500.0, y:400.0},
            DronePos{id:3, x:800.0, y:100.0},
        ];
        //MyDummy {
        //    drones: drone_poses,
        //}
        MyDummy {
            //drone: drone_poses[0].clone(),
            val:55,
        }
    }
}

#[wasm_bindgen]
pub fn plot() -> MyDummy {
    let mut my_dummy = MyDummy::new();
    let mynum:u32 = 1;
    my_dummy
    //serde_wasm_bindgen::to_value(&my_dummy).unwrap()
    //serde_wasm_bindgen::to_value(&my_dummy.drone).unwrap()
}


/*
#[derive(Clone, Copy)]
pub struct DroneData {
    pub id: usize,
    pub x: f32,
    pub y: f32,
}


pub struct DroneTelemetry{
    pub coords: Vec<DroneData>,
}

impl DroneTelemetry{
    pub fn new() -> DroneTelemetry {

        let coord_scale:f32 = 1./1000.;
        let mut coords:Vec<DroneData> = Vec::new();
        coords.push(DroneData{id:1, x:200.0*coord_scale, y:100.0*coord_scale});
        coords.push(DroneData{id:2, x:500.0*coord_scale, y:400.0*coord_scale});
        coords.push(DroneData{id:3, x:800.0*coord_scale, y:100.0*coord_scale});
        DroneTelemetry{coords: coords}
    }

    pub fn get_drone_positions(&self)->Vec<JsValue>{
        vec![JsValue::from_f64(self.coords[0].x as f64), JsValue::from_f64(self.coords[0].y as f64)]
    }
}
*/



