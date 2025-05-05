package com.travelplanner.travelplanner.model;

import jakarta.persistence.*;
import lombok.Data;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.List;

@Data
public class HotelDTO {

    private  Long id;
    private String name;
    private String address;
    private String description;
    private int stars;
    private double price;
    private List<PhotoDTO> photos;
    private List<ReviewDTO> reviews;
    private List<FacilityDTO> facilities;

}

