use drone_lib::{Position, Vec2};

use crate::types::Point;

/// World scale configuration for coordinate translation.
///
/// The frontend works in pixels (canvas coordinates), while drone-lib
/// works in meters (real-world SI units). This struct handles the
/// conversion between the two coordinate systems.
#[derive(Debug, Clone, Copy)]
pub struct WorldScale {
    /// Pixels per meter (e.g., 2.0 means 500m = 1000px)
    pub px_per_meter: f32,
    /// World width in meters
    pub world_width_meters: f32,
    /// World height in meters
    pub world_height_meters: f32,
    /// Canvas width in pixels
    pub canvas_width_px: f32,
    /// Canvas height in pixels
    pub canvas_height_px: f32,
}

impl WorldScale {
    /// Create from canvas size (pixels) and world size (meters).
    ///
    /// Uses the smaller ratio to ensure the world fits in the canvas.
    pub fn new(
        canvas_width_px: f32,
        canvas_height_px: f32,
        world_width_meters: f32,
        world_height_meters: f32,
    ) -> Self {
        // Calculate scale (use the smaller ratio to ensure world fits)
        let scale_x = canvas_width_px / world_width_meters;
        let scale_y = canvas_height_px / world_height_meters;
        let px_per_meter = scale_x.min(scale_y);

        WorldScale {
            px_per_meter,
            world_width_meters,
            world_height_meters,
            canvas_width_px,
            canvas_height_px,
        }
    }

    /// Default scale: 1000x1000 pixels = 2500x2500 meters (0.4 px/m)
    pub fn default_scale() -> Self {
        Self::new(1000.0, 1000.0, 2500.0, 2500.0)
    }

    /// Convert pixel distance to meters.
    #[inline]
    pub fn px_to_meters(&self, px: f32) -> f32 {
        px / self.px_per_meter
    }

    /// Convert meters to pixel distance.
    #[inline]
    pub fn meters_to_px(&self, meters: f32) -> f32 {
        meters * self.px_per_meter
    }

    /// Convert a Point from pixels to meters.
    pub fn point_px_to_meters(&self, p: Point) -> Point {
        Point {
            x: self.px_to_meters(p.x),
            y: self.px_to_meters(p.y),
        }
    }

    /// Convert a Point from meters to pixels.
    pub fn point_meters_to_px(&self, p: Point) -> Point {
        Point {
            x: self.meters_to_px(p.x),
            y: self.meters_to_px(p.y),
        }
    }

    /// Convert Position (meters) to Point (pixels).
    pub fn position_to_point_px(&self, p: Position) -> Point {
        Point {
            x: self.meters_to_px(p.x()),
            y: self.meters_to_px(p.y()),
        }
    }

    /// Convert Point (pixels) to Position (meters).
    pub fn point_px_to_position(&self, p: Point) -> Position {
        Position::new(self.px_to_meters(p.x), self.px_to_meters(p.y))
    }

    /// Convert Vec2 (meters) to Point (pixels).
    pub fn vec2_to_point_px(&self, v: Vec2) -> Point {
        Point {
            x: self.meters_to_px(v.x),
            y: self.meters_to_px(v.y),
        }
    }
}
