package com.travelplanner.travelplanner.model;

import jakarta.persistence.*;
import lombok.Data;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.List;

@Data
public class FacilityDTO {

    private  Long id;
    private List<HotelDTO> hotel;
    private String name ;

}
