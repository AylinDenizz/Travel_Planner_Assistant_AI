package com.travelplanner.travelplanner.model;

import jakarta.persistence.*;
import lombok.Data;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.sql.Timestamp;
import java.util.List;

@Data
public class ReviewDTO {

    private  Long id;
    private List<HotelDTO> hotel;
    private  String userName;
    private Double rating;
    private String comment;
    private Timestamp createdAt;


}
